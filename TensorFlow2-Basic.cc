/* Copyright 2023. All Rights Reserved.

Licensed und the Apache Licence, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law of agreed to in writing, software
distrubuted unde the License is distrubuted on an "AS IS" Basis,
WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/
/*
 This is a C++ version of the TensorFlow Basic Tutorial
 originally done in python

Compiled using:
export TF_CPP_MIN_LOG_LEVEL=2
nvcc TensorFlow2-Basic.cc -O3 -I/usr/local/lib/python3.8/dist-packages/tensorflow/include -L/usr/local/lib/python3.8/dist-packages/tensorflow -ltensorflow_cc -ltensorflow_framework -o TensorFlow2-basic
*/

//#define DEBUG
#include <chrono>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <mutex>
#include <tuple>
#include <tensorflow/cc/client/client_session.h>
#include "absl/synchronization/barrier.h"
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"
#include "tensorflow/core/util/work_sharder.h"
//#include "tensorflow/core/gradients/grad_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

// Quadratic function to be determined
// f(x)=x*x+2*x-5
//
Output f(const Scope &scope, Input x){
  return Subtract(scope,
        Add(scope,
           Square(scope,x),
           Multiply(scope, 2.f, x)
        ),5.f
      );
};

// Generates evenly-spaced values in an interval along the first axis.
// C++ API does not have the python Linspace equivalent.
//
class Linspace{
  const float m_minval;
  const float m_maxval;
  int m_num;
  std::vector<float> m_x;
public:
  Linspace(const float minval, const float maxval, int num)
    :m_minval(minval),
    m_maxval(maxval),
    m_num(num)
  {
     std::vector<float> incremental(m_num);
     m_x.resize(m_num);
     std::iota(incremental.begin(),incremental.end(),0.f);
     std::transform(incremental.begin(), incremental.end(), m_x.begin(),
       [this](float x_inc){
         return (1.f/float(this->m_num-1))*(this->m_maxval-this->m_minval)*x_inc+this->m_minval;
       }
     );

  };
  void operator()(Tensor input){
    std::copy_n(m_x.begin(), m_x.size(), input.flat<float>().data());
  };
};

// Generate random numbers
// used to initiallize the weights
//
class Rand{
    const Scope& m_scope;
    float m_minval{0.f};
    float m_maxval{5.f};
    int m_seed{22};
    Output m_rand_init;

  public:
    Rand(const Scope &scope)
      :m_scope(scope),
       m_rand_init(RandomUniform(m_scope, Input({1,3}), DT_FLOAT))
    {
       RandomUniform::Seed(m_seed);
    };
    const Output operator()(){
      auto tmp{Multiply(m_scope, Input(m_maxval-m_minval), Input(m_rand_init))};
      return Add(m_scope, tmp, Input(m_minval));
    };
};

// Batches the data
//
std::vector<Tensor> BatchSlice(int batch_size, const Input x)
{
  //Setup constants
  const auto n{x.tensor().tensor<float,1>().size()};
  const int batch_n{int(std::ceil(float(n)/float(batch_size)))};

  //Setup Destination list of Tensors
  std::vector<Tensor> x_sliced;
  for(int i=0; i<batch_n-1; ++i)
     x_sliced.push_back(Tensor(DT_FLOAT,{batch_size}));
  x_sliced.push_back(Tensor(DT_FLOAT,{n%batch_size}));

  //Slice to Destination Tensors
  for( int i=0; i<batch_n; ++i){

    TF_CHECK_OK(batch_util::CopyContiguousSlices(
      x.tensor(),//const Tensor & src,
      i*batch_size,//int64_t src_offset,
      0,//int64_t dst_offset,
      i<batch_n-1 ? batch_size : n % batch_size,//int64_t num_slices,
      &x_sliced[i]//Tensor *dst
    ));

  }
  return x_sliced;
};

// Shuffles the data
//
std::tuple<Tensor, Tensor> Shuffle(std::tuple<Tensor, Tensor> x)
{
  //Setup constants
  const long int n{std::get<0>(x).tensor<float,1>().size()};

  //Set up vector of tuples points
  std::vector<std::tuple< float, float>> pts;
  for(int i=0; i<n; ++i)
    pts.push_back( std::tuple<float, float>{ std::get<0>(x).tensor<float,1>()(i), std::get<1>(x).tensor<float,1>()(i)});

  //Randomize the Elements within the vector points.
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(pts.begin(), pts.end(), g);

  // Split the points tuple up into the tensors
  Tensor t0(DT_FLOAT,{n});
  Tensor t1(DT_FLOAT,{n});
  for(int i=0; i<n; ++i){
    t0.tensor<float,1>()(i)=std::get<0>(pts[i]);
    t1.tensor<float,1>()(i)=std::get<1>(pts[i]);
  }

  return std::tuple<Tensor, Tensor>{t0, t1};
};

// Class of the Quadratic model.
//
class Model{
  public:
    Tensor m_weights;
    std::mutex m_weights_mutex;

  public:
    Model(Tensor rand)
    {
       auto scope{ Scope::NewRootScope()};
       std::vector<Tensor> outputs;
       ClientSession session(scope, SessionOptions());
       TF_CHECK_OK(session.Run({ Rand(scope)()}, &outputs));
       assert(m_weights.CopyFrom( outputs[0], TensorShape({1,3})));
    };

    // Weighted quadradic
    // used in model output
    //
    Tensor operator()(Input x){

      auto scope{ Scope::NewRootScope()};

      auto tmp_ops {Stack(scope, { Square(scope, x.tensor()), x, Fill(scope, {x.tensor().tensor<float,1>().size()}, Input(1.f))})};

      Output tmp2_ops;
      {
        std::lock_guard<std::mutex> guard(m_weights_mutex);
        tmp2_ops=MatMul( scope, m_weights, tmp_ops);
      }

      std::vector<Tensor> outputs;
      ClientSession session(scope, SessionOptions());
      TF_CHECK_OK(session.Run({tmp2_ops}, &outputs));

      return outputs[0];
   }

   // Gradient of: Means Square Error(MSE) of the quadradic function.
   // Hand crafted instead of using the gradient tape method.
   // Main function used in the training.
   //
   Tensor MSEGradient(const Input& x, const Input& y){
#ifdef DEBUG
     std::cout << "Calculating MSEGradient -- Start " << std::endl;
#endif
     std::vector<Tensor> outputs;
     auto scope{ Scope::NewRootScope()};
     ClientSession session(scope, SessionOptions());

     const auto& x_pow0_ops{ Fill(scope, {x.tensor().tensor<float,1>().size()}, Input(1.f))};
     const auto& x_pow1_ops{ Multiply(scope, x.tensor(), Input(1.f))};
     const auto& x_pow2_ops{ Square(scope, x.tensor())};
     const auto& x_pow3_ops{ Multiply(scope, x.tensor(), x_pow2_ops)};
     const auto& x_pow4_ops{ Square(scope, x_pow2_ops)};
     const auto& x_pow2_y_ops{ Multiply(scope, x_pow2_ops, y.tensor())};
     const auto& x_y_ops{ Multiply(scope, x.tensor(), y.tensor())};
     const auto& y_ops{ Multiply(scope, y.tensor(), Input(1.f))};

     OutputList ops(7);

     const auto& t0=Input({{2.f, 2.f, 2.f, -2.f}});

     Output t1;
     {
       std::lock_guard<std::mutex> guard(m_weights_mutex);
       t1=Concat( scope, { m_weights, {{1.f}}}, 1);
     }

     ops[0]=Stack( scope, OutputList { x_pow4_ops, x_pow3_ops, x_pow2_ops, x_pow2_y_ops});
     ops[1]=Stack( scope, OutputList { x_pow3_ops, x_pow2_ops, x_pow1_ops, x_y_ops} );
     ops[2]=Stack( scope, OutputList { x_pow2_ops, x_pow1_ops, x_pow0_ops, y_ops });

     ops[3]=Multiply( scope, t0, t1);
     ops[4]=Mean( scope, MatMul( scope, ops[3], ops[0]), {1});
     ops[5]=Mean( scope, MatMul( scope, ops[3], ops[1]), {1});
     ops[6]=Mean( scope, MatMul( scope, ops[3], ops[2]), {1});

     Output out_ops;
     out_ops=Stack( scope, OutputList {ops[4], ops[5], ops[6]}, Stack::Axis(1));

#ifdef DEBUG
       if (!scope.ok()) {
         LOG(FATAL) << scope.status().ToString();
         abort();
       }
#endif
     TF_CHECK_OK(session.Run({out_ops}, &outputs));
#ifdef DEBUG
     std::cout << "Calculating MSEGradient -- Finish " << std::endl;
#endif
     return outputs[0];
  };

};

// Mean Square Error loss function
// Note Gradient already encapsulating this function.
// Used to output the MSE_Loss.
//
Output mse_loss(const Scope& scope, const Input& y_pred, const Input& y){

  //Change the vector to a matrix
  auto n{y.tensor().tensor<float,1>().size()};
  Tensor y_matrix(DT_FLOAT, {1, n});
  assert( y_matrix.CopyFrom(y.tensor(), TensorShape({n})));

  //Return the MSE
  return Mean(scope, Square(scope, Subtract(scope, y_pred, y_matrix)), {0});
};

// Main class to train the model.
// A functional for threading purposes.
//
class Train
{
  const float m_learning_rate;
  Model &m_model;
  const Tensor m_x;
  const Tensor m_y;

public:
  Train (const float learning_rate,
    Model& model,
    const Tensor x,
    const Tensor y
  ):m_learning_rate(learning_rate),
    m_model(model),
    m_x(x),
    m_y(y)
  {}

  void operator()(){
#ifdef DEBUG
    std::cout << "Entering training fn" << std::endl;
#endif

     auto scope{ Scope::NewRootScope()};

    //update parameters with respect to the gradient calculations

    auto grad{m_model.MSEGradient(m_x, m_y)};
    const auto& learning_step_ops{ Multiply(scope, m_learning_rate, grad)};

    Output new_weights;
    {
      std::lock_guard<std::mutex> guard(m_model.m_weights_mutex);
      new_weights=Sub(scope, m_model.m_weights, learning_step_ops);
    }

    std::vector<Tensor> outputs;
    ClientSession session(scope, SessionOptions());

#ifdef DEBUG
       if (!scope.ok()) {
         LOG(FATAL) << scope.status().ToString();
         abort();
       }
#endif
    TF_CHECK_OK(session.Run({new_weights}, &outputs));
    {
      std::lock_guard<std::mutex> guard(m_model.m_weights_mutex);
      assert(m_model.m_weights.CopyFrom( outputs[0], TensorShape({1,3})));
#ifdef DEBUG
      std::cout << m_model.m_weights.DebugString() << std::endl;
#endif
    }
  }
};

int main(int argc, char** argv){

  auto start = std::chrono::high_resolution_clock::now();

  //create a root scope
  auto scope {Scope::NewRootScope()};

  //create a session that takes our scope as the root scope
  ClientSession session(scope, SessionOptions());
  std::vector<Tensor> outputs;

  //Example Data
  //Generate Linspace
#ifdef DEBUG
  std::cout << "Generating Linspace" << std::endl;
#endif
  int num{201};
  Tensor x(DT_FLOAT,{num});
  Linspace(-2.f, +2.f, num)(x);
  auto y_ops {Add(scope, f(scope, x), RandomNormal(scope, {num}, DT_FLOAT))};
  TF_CHECK_OK(session.Run({y_ops}, &outputs));
  auto y {outputs[0]};

  //Print out the input numbers
  std::cout << "Input" << std::endl;
  for(int i=0; i<num; ++i)
    std::cout  << x.tensor<float,1>()(i) << ", " << y.tensor<float,1>()(i) << std::endl;

  //Generate the random numbers
  auto rand_input {Rand(scope)()};
  TF_CHECK_OK(session.Run({rand_input}, &outputs));

  //Generate the weights and bias
  Model quad_model(outputs[0]);

  //Set Batch and Shuffle
  auto x_batches=BatchSlice(32, x);
  auto y_batches=BatchSlice(32, y);
  std::vector<Tensor> x_random(x_batches.size()), y_random(y_batches.size());
  for(int i=0; i< x_batches.size(); ++i){
    std::tie(x_random[i], y_random[i])=Shuffle(std::tuple<Tensor, Tensor>{x_batches[i],y_batches[i]});
  }

  std::list<int> batch_index(x_random.size());
  std::iota(batch_index.begin(), batch_index.end(), 0);

  //Train
#ifdef DEBUG
  std::cout << "Training" << std::endl;
#endif

  int epochs{100};
  float learning_rate{0.01f};
  for(int epoch=0; epoch<epochs; ++epoch){
//    auto epoch_start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
    std::cout << "Begin for_each loop" << std::endl;
#endif

      std::for_each(batch_index.begin(), batch_index.end(),
        [&quad_model, learning_rate, x_random, y_random](const int &batch_i){
          Train train(learning_rate, quad_model, x_random[batch_i], y_random[batch_i]);
          train();
        }
      );


    if(epoch % 10 == 0){
#ifdef DEBUG
       if (!scope.ok()) {
         LOG(FATAL) << scope.status().ToString();
         abort();
       }
#endif

      TF_CHECK_OK(session.Run({mse_loss(scope, quad_model(x), y)}, &outputs));
      std::cout << "Mean squared error for step " << epoch << ": " << outputs[0].tensor<float,1>()(0);
      {
        std::lock_guard<std::mutex> guard(quad_model.m_weights_mutex);
        std::cout << " Weights: " << quad_model.m_weights.tensor<float,2>()(0,0);
        std::cout << ", " << quad_model.m_weights.tensor<float,2>()(0,1);
        std::cout << ", " << quad_model.m_weights.tensor<float,2>()(0,2) << std::endl;
      }

    }
//    auto epoch_end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
//    auto epoch_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(epoch_duration);
//    std::cout << "Epoch Time taken: " << epoch_microseconds.count() << " microseconds" << std::endl;
  }

  //Calc the predictions
  std::cout << "Calc Predictions" << std::endl;
  auto y_pred = quad_model(x);
  for(int i=0; i<x.tensor<float,1>().size(); ++i){
    std::cout << x.tensor<float,1>()(i);
    std::cout << ", " << y_pred.tensor<float,2>()(0,i);
    std::cout << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;

  // Convert duration to microseconds for better resolution
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);

  // Output the duration in microseconds
  std::cout << "Time taken: " << microseconds.count() << " microseconds" << std::endl;

}//end of main

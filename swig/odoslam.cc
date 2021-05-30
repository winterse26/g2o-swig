// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "odoslam.h"

#include <string>
#include <cassert>

#include "g2o/config.h"
#include "g2o/core/factory.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/hyper_dijkstra.h"
#include "g2o/core/optimization_algorithm.h"
#include "g2o/core/sparse_optimizer_terminate_action.h"

#include "g2o/stuff/macros.h"

#include "graph.pb.h"

static bool hasToStop = false;
static double gain = 1e-6;

using namespace g2o;
using namespace std;

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(csparse);
G2O_USE_TYPE_GROUP(slam3d);
G2O_USE_TYPE_GROUP(slam3d_addons);

int process(unsigned char *buf, int size, int capacity, int *osize, int maxIterations, bool useGain, bool checkInput, bool verbose) {

  OptimizableGraph::initMultiThreading();

  if (verbose) {
    cout << "# Used Compiler: " << G2O_CXX_COMPILER << endl;
  }

  SparseOptimizer optimizer;
  OptimizationAlgorithmProperty solverProperty;

  const string strSolver = "lm_var";

  optimizer.setVerbose(false);
  optimizer.setForceStopFlag(&hasToStop);

  if (useGain) {
    if (verbose) {
      cerr << "# Setup termination criterion based on the gain of the iteration" << endl;
    }
    SparseOptimizerTerminateAction* terminateAction = new SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(gain);
    terminateAction->setMaxIterations(maxIterations);
    optimizer.addPostIterationAction(terminateAction);
  }

  OptimizationAlgorithmFactory* const solverFactory = OptimizationAlgorithmFactory::instance();
  optimizer.setAlgorithm(solverFactory->construct(strSolver, solverProperty));

  if (!optimizer.solver()) {
    cerr << "Error allocating solver. Allocating " << strSolver << " failed!" << endl;
    return 1;
  }

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (buf != NULL) {
    if (verbose) {
      cerr << "# Loaded " << size << " bytes" << endl;
    }

    google::protobuf::io::ArrayInputStream arr(buf, size);

    g2o::proto::Graph graph;
    if (!graph.ParseFromZeroCopyStream(&arr)) {
      cerr << "Failed to parse proto graph." << endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 2;
    }

    if (!optimizer.loadProto(graph)) {
      cerr << "Error loading graph" << endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 2;
    }

    if (verbose) {
      cerr << "# Loaded " << graph.row_size() << " rows" << endl;
      cerr << "# Loaded " << optimizer.vertices().size() << " vertices" << endl;
      cerr << "# Loaded " << optimizer.edges().size() << " edges" << endl;
    }

    if (optimizer.vertices().size() == 0) {
      cerr << "Graph contains no vertices" << endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 3;
    }

    const set<int> vertexDimensions = optimizer.dimensions();
    if (!optimizer.isSolverSuitable(solverProperty, vertexDimensions)) {
      cerr << "The selected solver is not suitable for optimizing the given graph" << endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
  }

  assert (optimizer.solver());

  if (checkInput) {
    bool gaugeFreedom = optimizer.gaugeFreedom();
    OptimizableGraph::Vertex* gauge = optimizer.findGauge();
    if (gaugeFreedom) {
      if (!gauge) {
        cerr << "Cannot find a vertex to fix in this thing" << endl;
        google::protobuf::ShutdownProtobufLibrary();
        return 3;
      } else {
        if (verbose) {
          cerr << "# Graph is fixed by node " << gauge->id() << endl;
        }
        gauge->setFixed(true);
      }
    } else if (verbose) {
      cerr << "# Graph is fixed by priors or already fixed vertex" << endl;
    }

    HyperDijkstra d(&optimizer);
    UniformCostFunction f;
    d.shortestPaths(gauge, &f);

    if (d.visited().size() != optimizer.vertices().size()) {
      cerr << "Warning: d.visited().size() != optimizer.vertices().size()" << endl;
      cerr << "Visited: " << d.visited().size() << endl;
      cerr << "Vertices: " << optimizer.vertices().size() << endl;
    }
  }

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  const double loadChi = optimizer.chi2();

  if (verbose) {
    cerr << "# Load  chi2 = " << FIXED(loadChi) << endl;
  }

  const int result = optimizer.optimize(maxIterations);
  if (maxIterations > 0 && result == OptimizationAlgorithm::Fail) {
    cerr << "Cholesky failed, result might be invalid" << endl;
  }

  optimizer.computeActiveErrors();
  const double finalChi = optimizer.chi2();
  if (verbose) {
    cerr << "# Final chi2 = " << FIXED(finalChi) << endl;
  }

  g2o::proto::Graph ograph;

  if (!optimizer.saveProto(ograph)) {
    cerr << "Error saving graph" << endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 2;
  }

  int nsize = ograph.ByteSizeLong();

  if (verbose) {
    cerr << "# Return " << ograph.row_size() << " rows" << endl;
    cerr << "# Return " << nsize << " bytes" << endl;
  }

  if (nsize > capacity) {
    cerr << "Input buffer capacity " << capacity << " too small for result " << nsize << endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 2;
  }

  bool success = ograph.SerializeToArray(buf, nsize);
  if (osize != NULL && success) {
    *osize = nsize;
  } else {
    if (osize != NULL) {
      *osize = 0;
    }
    cerr << "Failed to write proto graph." << endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 2;
  }

  google::protobuf::ShutdownProtobufLibrary();

  // TODO
  //Destroy all the singletons
  //Factory::destroy();
  //OptimizationAlgorithmFactory::destroy();
  //HyperGraphActionLibrary::destroy();

  return 0;
}

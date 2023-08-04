//
// Created by 郑毓嘉 on 2023/8/3.
//
#ifndef BENCHMARK_GENERATOR_PASS_BENCHMARKGENERATOR_H
#define BENCHMARK_GENERATOR_PASS_BENCHMARKGENERATOR_H
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {
    class BenchMarkGenerator : public PassInfoMixin<BenchMarkGenerator> {
    public:
        PreservedAnalyses run(Function& f, FunctionAnalysisManager& fam);
    };
}
#endif //BENCHMARK_GENERATOR_PASS_BENCHMARKGENERATOR_H

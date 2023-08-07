//
// Created by 郑毓嘉 on 2023/8/3.
//
#include "BenchMarkGenerator.h"
#include<vector>
#include<set>
#include<sstream>
#include<fstream>
#include<filesystem>

namespace fs = std::filesystem;


using namespace llvm;

namespace{
    std::string part1 = "#include<cuda.h>\n"
                        "#include<iostream>\n"
                        "#include<time.h>\n"
                        "#include<stdio.h>\n"
                        "\n"
                        "static void check(CUresult result, char const *const func,\n"
                        "                  const char *const file, int const line) {\n"
                        "    if (result) {\n"
                        "        fprintf(stderr, \"CUDA error at %s:%d code=%d \\\"%s\\\" \\n\", file, line,\n"
                        "                static_cast<unsigned int>(result), func);\n"
                        "        exit(EXIT_FAILURE);\n"
                        "    }\n"
                        "}\n"
                        "\n"
                        "#define checkCudaDrvErrors(val) check((val), #val, __FILE__, __LINE__)\n"
                        "\n"
                        "int main(int argc, char** argv)\n"
                        "{\n"
                        "    CUdevice device;\n"
                        "    CUfunction func;\n"
                        "    CUmodule module;\n"
                        "    CUcontext ctx;\n"
                        "\n"
                        "    double begin, end;\n"
                        "\n"
                        "    checkCudaDrvErrors(cuInit(0));\n"
                        "\n"
                        "    checkCudaDrvErrors(cuDeviceGet(&device, 0));\n"
                        "    checkCudaDrvErrors(cuCtxCreate(&ctx, 0, device));\n";

    std::string part2 = "    begin = clock();\n"
                        "    cuLaunchKernel(func, grid_x, grid_y, grid_z, thread_x, thread_y, thread_z, share_memory_byte,stream, paras, extra);\n"
                        "    end = clock();\n"
                        "\n"
                        "    std::cout << end - begin << std::endl;\n"
                        "}";


    uint32_t get_module_start(const std::string& s)
    {
        for(int i=s.size()-1; i>=1; i--)
        {
            if(s[i] == '/' && s[i-1] == '.')
                return i+1;
        }
        return 0;
    }

    std::string get_module_name(StringRef path)
    {
        std::string p = path.str();
        uint32_t i = get_module_start(p);
        return p.substr(i, p.size()-i-3);
    }

    std::string code_gen(Function& f, const std::string& module_name, std::vector<uint64_t>& parameter_bytes, uint32_t share_memory_byte,
                  uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                  uint32_t thread_x, uint32_t thread_y, uint32_t thread_z)
    {
        std::stringstream code;
        code << part1;
        code << "    checkCudaDrvErrors(cuModuleLoad(&module, \""<<module_name<<".cubin\"));\n";
        code << "    checkCudaDrvErrors(cuModuleGetFunction(&func, module, " << "\""<<f.getName().str()<<"\"));\n";
        for(int i=0; i<parameter_bytes.size(); i++)
            code << "    CUdeviceptr p"<<i<<";\n";

        for(int i=0; i<parameter_bytes.size(); i++)
            code << "    checkCudaDrvErrors(" << "cuMemAlloc(" <<"&p"<<i <<", "<<parameter_bytes[i]<<")"  <<");\n";

        code<<"    int grid_x = "<<grid_x<<";\n";
        code<<"    int grid_y = "<<grid_y<<";\n";
        code<<"    int grid_z = "<<grid_z<<";\n";
        code<<"    int thread_x = "<<thread_x<<";\n";
        code<<"    int thread_y = "<<thread_y<<";\n";
        code<<"    int thread_z = "<<thread_z<<";\n";

        code<<"    int share_memory_byte = "<<share_memory_byte<<";\n";
        code<<"    CUstream stream = 0;\n";
        code<<"    void** extra = 0;\n";

        code<<"    void* paras[] = {";
        for(int i=0; i<parameter_bytes.size(); i++)
        {
            code << "&p" <<i;
            if(i != parameter_bytes.size()-1)
                code<<", ";
        }
        code<<"};\n";

        code<<part2<<"\n";
        return code.str();
    }
}


void byte_of_parameters(Function &f, std::vector<uint64_t>& bytes)
{
    for(const auto& arg: f.args())
    {
        if(!arg.getType()->isPointerTy())
        {
            outs()<<f.getName()<<" has no ptr argument \n";
            exit(1);
        }
        bytes.push_back(arg.getDereferenceableBytes());
        //outs()<<arg.getDereferenceableBytes()<<"\n";
    }
}

void get_Intrinsics(Function &f, std::vector<CallInst*>& intrinsic_call)
{
    for(auto& inst: instructions(f))
    {
        auto call_inst = dyn_cast<CallInst>(&inst);
        if(call_inst)
        {
            if(call_inst->getCalledFunction()->isIntrinsic())
            {
                Intrinsic::ID id = call_inst->getCalledFunction()->getIntrinsicID();

                if(id == Intrinsic::nvvm_read_ptx_sreg_tid_x
                || id == Intrinsic::nvvm_read_ptx_sreg_tid_y
                || id == Intrinsic::nvvm_read_ptx_sreg_tid_z
                || id == Intrinsic::nvvm_read_ptx_sreg_ctaid_x
                || id == Intrinsic::nvvm_read_ptx_sreg_ctaid_y
                || id == Intrinsic::nvvm_read_ptx_sreg_ctaid_z)
                    intrinsic_call.push_back(call_inst);
            }
        }
    }
}

uint32_t get_upper_bound_of(MDNode* node)
{
    const MDOperand& upper = node->getOperand(1);
    const Metadata* data = upper.get();
    const ValueAsMetadata* vd = dyn_cast<ValueAsMetadata>(data);
    const ConstantInt* ci = dyn_cast<ConstantInt>(vd->getValue());

    if(ci)
        return ci->getZExtValue();
    else
    {
        errs()<<"in get_upper_bound_of\n";
        exit(1);
    }
}

void write_to_benchmark_file(Function& f, const std::string& code)
{
    std::string dir_path = "./benchmarks";
    if(!fs::is_directory(dir_path))
    {
        if(!fs::create_directory(dir_path))
        {
            errs()<<"can't not create benchmark directory\n";
            exit(1);
        }
    }
    std::stringstream benchmark_name;
    benchmark_name<<dir_path<<"/";
    benchmark_name << f.getName().str()<< "_benchmark.cpp";
    outs()<<benchmark_name.str()<<"\n";
    std::ofstream of;
    of.open(benchmark_name.str(), std::ios::out);
    of<<code;
    of.close();
}

PreservedAnalyses BenchMarkGenerator::run(llvm::Function &f, llvm::FunctionAnalysisManager &fam) {
    std::vector<uint64_t> bytes;
    std::vector<CallInst*> intrinsics;
    byte_of_parameters(f, bytes);

    get_Intrinsics(f, intrinsics);
    std::set<Intrinsic::ID> seen;
    uint32_t block_x = 1;
    uint32_t block_y = 1;
    uint32_t block_z = 1;
    uint32_t thread_x = 1;
    uint32_t thread_y = 1;
    uint32_t thread_z = 1;

    for(const auto* intrin: intrinsics)
    {
        if(seen.find(intrin->getIntrinsicID()) == seen.end())
            seen.insert(intrin->getIntrinsicID());
        else
        {
            errs()<<"multiple intrin found\n";
            exit(1);
        }

        Intrinsic::ID id = intrin->getCalledFunction()->getIntrinsicID();
        MDNode* range = intrin->getMetadata("range");
        uint32_t upper_bound = get_upper_bound_of(range);

        switch (id) {
            case Intrinsic::nvvm_read_ptx_sreg_tid_x:
                thread_x = upper_bound;
                break;
            case Intrinsic::nvvm_read_ptx_sreg_tid_y:
                thread_y = upper_bound;
                break;
            case Intrinsic::nvvm_read_ptx_sreg_tid_z:
                thread_z = upper_bound;
                break;
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
                block_x = upper_bound;
                break;
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
                block_x = upper_bound;
                break;
            case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
                block_x = upper_bound;
                break;
        }
    }
    std::string module_name = get_module_name(f.getParent()->getName());
    std::string code = code_gen(f, module_name, bytes, 0, block_x, block_y, block_z, thread_x, thread_y, thread_z);
    write_to_benchmark_file(f, code);
    return PreservedAnalyses::all();
}


extern "C" PassPluginLibraryInfo
llvmGetPassPluginInfo()
{
    return {
      LLVM_PLUGIN_API_VERSION, "BenchMarkGenerator", "v0.1",
      [](PassBuilder& pb){
          using PipelineElement = typename PassBuilder::PipelineElement;
          pb.registerPipelineParsingCallback([](StringRef name, FunctionPassManager& fpm, ArrayRef<PipelineElement> eles){
              if(name == "BenchMark-Generator")
              {
                  fpm.addPass(BenchMarkGenerator());
                  return true;
              }
              return false;
          });
      }
    };
}

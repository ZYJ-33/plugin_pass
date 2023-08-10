//
// Created by 郑毓嘉 on 2023/8/3.
//
#include "BenchMarkGenerator.h"
#include<vector>
#include<set>
#include<map>
#include<sstream>
#include<fstream>
#include<sys/stat.h>
#include<sys/types.h>

using namespace llvm;

namespace{
    std::string part1 = "#include<cuda.h>\n"
                        "#include<cuda_runtime.h>\n"
                        "#include<iostream>\n"
                        "#include<time.h>\n"
                        "#include<stdio.h>\n"
                        "#include<bits/stdc++.h>\n"
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
                        "    cudaEvent_t begin, end;\n"
                        "    cudaEventCreate(&begin);\n"
                        "    cudaEventCreate(&end);\n"
                        "\n"
                        "    checkCudaDrvErrors(cuInit(0));\n"
                        "\n"
                        "    checkCudaDrvErrors(cuDeviceGet(&device, 0));\n"
                        "    checkCudaDrvErrors(cuCtxCreate(&ctx, 0, device));\n";



    std::string part2 = "    cudaEventRecord(begin);\n"
                        "    cuLaunchKernel(func, grid_x, grid_y, grid_z, thread_x, thread_y, thread_z, share_memory_byte,stream, paras, extra);\n"
                        "    cudaEventRecord(end);\n"
                        "    cudaEventSynchronize(end);\n"
                        "    float time;\n"
                        "    cudaEventElapsedTime(&time, begin, end);"

                        "\n"
                        "    std::cout << time << std::endl;\n"
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

    std::string code_gen(Function& f, const std::string& module_name, std::vector<uint64_t>& parameter_bytes, uint64_t share_memory_byte,
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

bool dir_exist(const std::string& path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

void create_dir(const std::string& path)
{
    if(mkdir(path.c_str(), 0777) != 0)
    {
        errs()<<"can't not create benchmark directory\n";
        exit(1);
    }
}

void write_to_benchmark_file(Function& f, const std::string& code)
{
    std::string dir_path = "./benchmarks";
    if(!dir_exist(dir_path))
        create_dir(dir_path);

    std::stringstream benchmark_name;
    benchmark_name<<dir_path<<"/";
    benchmark_name << f.getName().str()<< "_benchmark.cpp";
    outs()<<benchmark_name.str()<<"\n";
    std::ofstream of;
    of.open(benchmark_name.str(), std::ios::out);
    of<<code;
    of.close();
}

bool is_load_or_store(const User* inst)
{
    const Instruction* load_or_store = dyn_cast<Instruction>(inst);
    if(load_or_store)
        return load_or_store->getOpcode() == Instruction::Load || load_or_store->getOpcode() == Instruction::Store;
    return false;
}

const Instruction* to_inst(const User* user)
{
    if(!user)
        return nullptr;

    const Instruction* inst = dyn_cast<Instruction>(user);
    if(inst)
        return inst;
    const Operator* op = dyn_cast<Operator>(user);
    if(!op)
        return nullptr;

    auto adsc = dyn_cast<ConcreteOperator<Operator, Instruction::AddrSpaceCast>>(op);
    auto gep = dyn_cast<ConcreteOperator<Operator, Instruction::GetElementPtr>>(op);
    if(!gep && !adsc)
        return nullptr;

    if(gep)
    {
        for(auto u : gep->users())
        {
            auto res = to_inst(u);
            if(res)
                return res;
        }
        return nullptr;
    }
    else
    {
        for(auto u : adsc->users())
        {
            auto res = to_inst(u);
            if(res)
                return res;
        }
        return nullptr;
    }
}

static bool sharemem_calculate_before = false;
static std::map<std::string, uint64_t> func_sharedmemory;


void get_function_sharememory_usage(const Module* mod)
{
    for(auto iter = mod->global_begin(); iter != mod->global_end(); iter++)
    {
        const GlobalVariable* var = dyn_cast<GlobalVariable>(dyn_cast<GlobalObject>(iter));
        bool can_be_null;
        bool can_be_free;
        auto size = var->getPointerDereferenceableBytes(mod->getDataLayout(),  can_be_null, can_be_free);

        std::set<std::string> func_record;
        if(var->getAddressSpace() == 3)
        {
            for(auto user_iter: var->users())
            {
                auto inst_ptr = to_inst(user_iter);
                if(!inst_ptr)
                {
                    errs()<<"in get_function_sharememory_usage\n"<<*user_iter<<"\n";
                    exit(1);
                }
                std::string func_name = inst_ptr->getParent()->getParent()->getName().str();
                if(func_record.find(func_name) != func_record.end())
                    continue;
                func_record.insert(func_name);

                if(func_sharedmemory.find(func_name) == func_sharedmemory.end())
                    func_sharedmemory[func_name] = size;
                else
                    func_sharedmemory[func_name] += size;
            }
        }
    }
}

PreservedAnalyses BenchMarkGenerator::run(llvm::Function &f, llvm::FunctionAnalysisManager &fam) {

    if(!sharemem_calculate_before)
    {
        sharemem_calculate_before = true;
        get_function_sharememory_usage(f.getParent());
    }

    uint64_t share_memory_byte = func_sharedmemory[f.getName().str()];

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
    std::string code = code_gen(f, module_name, bytes, share_memory_byte, block_x, block_y, block_z, thread_x, thread_y, thread_z);
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

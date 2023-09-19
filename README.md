# README.md

### 总述

- 本工作是 [NVIDIA TensorRT Hackathon 2023](<u>[https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023)</u>) 的参赛题目，我们采用的选题是 3+4，即在现有模型 llama 的基础上进行优化
- 我们的优化效果如下所示：

时间：

```json
TensorRT-LLM (total latency: 67.72877144813538 sec)
```

精度：

- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在 Docker 里面代码编译、运行步骤：

我们使用的编译指令：

```json
pip install -e .
./scripts/build_wheel.py --trt_root --clean /usr/local/TensorRT-9.0.0.2
```

使用的 engine 编译指令：

在/root/NVIDIA_TensorRT_Hackathon_2023_Rematch/tensorrt_llm_july-release-v1/examples/llama 下运行

```json
python3 build.py --model_dir ./tmp/llama/7B/ \
                    --dtype float16 \
                    --use_gpt_attention_plugin float16 \
                    --use_gemm_plugin float16 \
                    --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/ \
                    --use_RMSnorm_plugin float32
```

编译 engine 后，运行 run.py 指令：

```json
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/
```

运行 summarize.py 指令：

```json
python summarize.py  --test_trt_llm \
                     --hf_model_location ./tmp/llama/7B/   \
                     --data_type fp16  \
                     --engine_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/
```

### 主要开发工作

#### 开发工作的难点

- 主要开发 rmsnorm 的 plugin，难点在于 cuda 算子代码的编写和 plugin 嵌入 build.py 的整个流程

### 开发与优化过程

在看 llama 的代码中发现，rmsnorm 部分只有简单的类似 pytorch 的推理实现，如下所示：

```json
with precision("float32"):
    varx = pow(input, 2.0)  # (-1, -1, 4096)
    varx = varx.mean(dim, keepdim=True)  # (-1, -1, 1)
    denom = varx + eps  # (-1,-1,1)
    denom = denom.sqrt()  # (-1, -1, 1)
    y = input / denom  # (-1, -1, 4096)

if weight is not None:
    y = y * weight  # (-1, -1, 4096)
```

这部分有很多可以并行的地方，因此考虑编写 plugin 来实现代码加速

**以下部分是 python 代码，主要是如何在 build engine 的过程中调用 plugin 进行编译：**

- 在 build.py 中编写接口，从而可以使用不同数据类型的 rmsnorm plugin

```json
if args.use_RMSnorm_plugin:
   print(args.use_RMSnorm_plugin)
   network.plugin_config.set_RMSnorm_plugin(dtype=args.use_RMSnorm_plugin)
```

- 在 plugin.py 中设置标志位，使得下一步中可以运行 plugin 部分

```json
def set_RMSnorm_plugin(self, dtype='float32'):
        self.RMSnorm_plugin = dtype
        return self
```

- 在 functional.py 中替换原始 rmsnorm 的写法，使得 rms_norm 函数调用对应的 rmsnorm plugin

```json
else:  # 加标志位运行这里
        plg_creator = trt.get_plugin_registry().get_plugin_creator('Rmsnorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                    trt.PluginFieldType.FLOAT32)
        p_dtype = default_net().plugin_config.RMSnorm_plugin
        pf_type = trt.PluginField("type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([eps, pf_type])
        rmsnorm_plug = plg_creator.create_plugin("rmsnorm", pfc)

        if weight is None:
            weight = constant(np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [input.trt_tensor, weight.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
        y =  _create_tensor(layer.get_output(0), layer)
    # ----------------------------------------------------------------
    return y
```

**以下部分是****C++****代码，主要是如何编写 rmsnorm 的 plugin**

- 在 cpp/tensorrt_llm/plugins/rmsnormPlugin 文件夹下创建 cpp 文件，创建 RmsnormPlugin 类，主要是 enquene 函数编写代码

```json
int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];
    if (mType == DataType::kHALF)       // 运行这里
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, mEps, m, n, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, mEps, m, n, stream);
    }
```

- 在 cpp/tensorrt_llm/kernels/创建 cuda 算子 rmsnormKernels.cu 文件，编写算子代码，核函数如下所示：

```json
template <typename T>
__global__ void generalRmsNorm(const T* input, const T* gamma, T* normed_output, const float eps,
    int tokens, int hidden_dim, const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    // __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    // float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }
        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        // local_sum += cuda_sum<float>(val_f);
        local_var_sum += cuda_sum<float>(val_f * val_f);
    }

    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
    {
        variance = (variance / hidden_dim); // Var[x] = E[x²] - E[x]²
        s_variance = rsqrtf(variance + eps);

    }
    __syncthreads();

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    // const bool with_beta = beta != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_rmsnorm(val_f, s_variance, gamma, i));
        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_rmsnorm(val_f, s_variance, gamma, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}
```

至此，rmsnorm plugin 完成。

### 优化效果

#### 云主机环境

**硬件环境**

- CPU Architecture: x86_64
- CPU Cores: 16
- CPU Model: Intel Xeon Platinum 8369B @ 2.90GHz
- NUMA Nodes: Single node with CPUs 0-15
- Total memory: 58GiB

**软件环境**

- Operating System: Ubuntu 22.04.2 LTS
- Docker Version: 24.0.5

#### 对比测试

我们的工作为针对 `llama` 的模型优化，以下是精度对比测试和性能加速测试的结果。

**精度**

在 tensorrt-LLM 中运行 build.py 和 summarize.py 后得到的精度如下所示：

```json
[08/31/2023-11:12:48] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/31/2023-11:12:48] [TRT-LLM] [I]   rouge1 : 20.762954367545632
[08/31/2023-11:12:48] [TRT-LLM] [I]   rouge2 : 5.683615121132041
[08/31/2023-11:12:48] [TRT-LLM] [I]   rougeL : 15.54620501769291
[08/31/2023-11:12:48] [TRT-LLM] [I]   rougeLsum : 18.385163805623428
```

在加入 rmsnorm plugin 后精度如下所示：

```json
[09/19/2023-13:11:32] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/19/2023-13:11:32] [TRT-LLM] [I]   rouge1 : 20.29824254208154
[09/19/2023-13:11:32] [TRT-LLM] [I]   rouge2 : 5.683615121132041
[09/19/2023-13:11:32] [TRT-LLM] [I]   rougeL : 15.38370548448208
[09/19/2023-13:11:32] [TRT-LLM] [I]   rougeLsum : 18.120947608991973
```

**性能**

在 tensorrt-LLM 中运行 build.py 和 summarize.py 后得到的时间如下所示：

```json
[09/19/2023-13:11:32] [TRT-LLM] [I] TensorRT-LLM (total latency: 67.993452693425 sec)
```

在加入 rmsnorm plugin 后时间如下所示：

```json
[09/19/2023-13:11:32] [TRT-LLM] [I] TensorRT-LLM (total latency: 67.611492395401 sec)
```

### 送分题答案

#### 送分题 1

运行命令

- python3 run.py --max_output_len=8

得到输出如下

```bash
root@6375fe14b223:~/workspace/tensorrt_llm_july-release-v1/examples/gpt# python3 run.py --max_output_len=8

Input: Born in north-east France, Soyer trained as a
Output:  chef and eventually became a chef at a
```

#### 送分题 2

运行命令

- python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf --batch_size1 --test_trt_llm --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14

得到输出如下

```bash
root@6375fe14b223:~/workspace/tensorrt_llm_july-release-v1/examples/gpt# python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu                      --test_hf                      --batch_size 1                      --test_trt_llm                      --hf_model_location=gpt2                      --check_accuracy                      --tensorrt_llm_rouge1_threshold=14

[08/18/2023-15:32:26] [TRT-LLM] [I] ---------------------------------------------------------
[08/18/2023-15:32:26] [TRT-LLM] [I] TensorRT-LLM Generated : 
[08/18/2023-15:32:26] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
[08/18/2023-15:32:26] [TRT-LLM] [I] 
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[08/18/2023-15:32:26] [TRT-LLM] [I] 
 Output : [[' Best died at age 88.']]
[08/18/2023-15:32:26] [TRT-LLM] [I] ---------------------------------------------------------
[08/18/2023-15:32:27] [TRT-LLM] [I] ---------------------------------------------------------
[08/18/2023-15:32:27] [TRT-LLM] [I] HF Generated : 
[08/18/2023-15:32:27] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
[08/18/2023-15:32:27] [TRT-LLM] [I] 
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[08/18/2023-15:32:27] [TRT-LLM] [I] 
 Output : [[' Best died at age 88.']]
[08/18/2023-15:32:27] [TRT-LLM] [I] ---------------------------------------------------------
Downloading builder script: 5.60kB [00:00, 15.7kB/s]                            
Token indices sequence length is longer than the specified maximum sequence length for this model (1151 > 1024). Running this sequence through the model will result in indexing errors
[08/18/2023-15:33:05] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.641429901123047 sec)
[08/18/2023-15:33:05] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/18/2023-15:33:05] [TRT-LLM] [I]   rouge1 : 15.361040799540035
[08/18/2023-15:33:05] [TRT-LLM] [I]   rouge2 : 3.854022269668396
[08/18/2023-15:33:05] [TRT-LLM] [I]   rougeL : 12.078455591738333
[08/18/2023-15:33:05] [TRT-LLM] [I]   rougeLsum : 13.547802733617264
[08/18/2023-15:33:05] [TRT-LLM] [I] Hugging Face (total latency: 10.247140169143677 sec)
[08/18/2023-15:33:05] [TRT-LLM] [I] HF beam 0 result
[08/18/2023-15:33:05] [TRT-LLM] [I]   rouge1 : 14.75593024343394
[08/18/2023-15:33:05] [TRT-LLM] [I]   rouge2 : 3.3647470801871733
[08/18/2023-15:33:05] [TRT-LLM] [I]   rougeL : 11.124766996533
[08/18/2023-15:33:05] [TRT-LLM] [I]   rougeLsum : 13.031128048110618
```

### 经验与体会

- 项目中生成大型文件较多，生成时间长，如何进行 Git 多人协作，如何保证大文件版本同步是一个难题
- 初次接触 tensorrt 和 cuda，挫折很多，比如一个 block 中最多只有 1024 个线程，在实战中也确实学习到不少知识。知道该如何编写 cuda 核函数，知道整个 plugin 流程是怎么样的，对 tensorrt 的认识更加深刻了。
- 对大语言模型的优化点有了一些了解。量化的作用，kv cache 优化这些词汇的理解更加深入了，想去尝试 AutoGPTQ 的量化导入，但由于时间来不及了，就没有完成。
- 非常感谢导师的耐心解答，感谢 nvidia 给予了我们一次学习的机会。

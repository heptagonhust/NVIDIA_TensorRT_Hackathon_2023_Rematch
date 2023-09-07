/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/plugins/rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::RmsnormPluginCreator;
using nvinfer1::plugin::RmsnormPlugin;

static const char* RMSNORM_PLUGIN_VERSION{"1"};
static const char* RMSNORM_PLUGIN_NAME{"Rmsnorm"};
PluginFieldCollection RmsnormPluginCreator::mFC{};
std::vector<PluginField> RmsnormPluginCreator::mPluginAttributes;

RmsnormPlugin::RmsnormPlugin(float eps)
    : mEps(eps)
{
}

// Parameterized constructor
RmsnormPlugin::RmsnormPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RmsnormPlugin::clone() const noexcept // 将这个plugin对象克隆一份给tensorrt的builder、network或者engine。
{
    auto* plugin = new RmsnormPlugin(mEps);  // 将要clone的plugin的权重和参数传递给这个构造函数
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;  // clone成员函数主要用于传递不变的权重和参数，将plugin复制n多份，从而可以被不同engine或者builder或者network使用
}

nvinfer1::DimsExprs RmsnormPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool RmsnormPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(0 <= pos && pos < 5);
    // return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    return true;
}

void RmsnormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,  // 配置这个插件，判断输入和输出类型数量是否正确。通过这个配置信息可以告知tensorrt去选择合适的算法去调优这个模型
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RmsnormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,  //返回这个插件需要中间显存变量的实际数据大小，通过tensorrt的接口去获取，是比较规范的方式
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsnormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, // 实现插件的执行函数，我们自己实现的cuda操作就放到这里，与往常一样接受输入inputs产生输出outputs，传给相应的指针就可以
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    //     bias [N, ]
    // outputs
    //     output [M(*), N]

    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    const half* input = reinterpret_cast<const half*>(inputs[0]);
    const half* weight = reinterpret_cast<const half*>(inputs[1]);
    half* output = reinterpret_cast<half*>(outputs[0]);
    invokeGeneralRmsNorm(output, input, weight, mEps, m, n, stream);
    // else if (mType == DataType::kFLOAT)
    // {
    //     const float* input = reinterpret_cast<const float*>(inputs[0]);
    //     const float* weight = reinterpret_cast<const float*>(inputs[1]);
    //     const float* bias = reinterpret_cast<const float*>(inputs[2]);
    //     float* output = reinterpret_cast<float*>(outputs[0]);
    //     invokeGeneralLayerNorm(output, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares);
    // }
#ifdef ENABLE_BF16
    // if (mType == DataType::kBF16)
    // {
    // const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(inputs[0]);
    // const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
    // __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
    // invokeGeneralRmsNorm(output, input, weight, mEps, m, n, stream);
    // }
#endif

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RmsnormPlugin::getOutputDataType(  // 返回结果的类型，一般来说插件返回结果类型与输入类型一致
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* RmsnormPlugin::getPluginType() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsnormPlugin::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

int RmsnormPlugin::getNbOutputs() const noexcept  //插件op返回多少个tensor。layernorm只输出一个tensor，所以直接return 1
{
    return 1;
}

int RmsnormPlugin::initialize() noexcept  // 初始化函数，在这个插件准备开始run之前执行。一般提前开辟参数的空间
{
    return 0;
}

void RmsnormPlugin::terminate() noexcept {}

size_t RmsnormPlugin::getSerializationSize() const noexcept  // 返回序列化时需要写多少个字节到buffer中。
{
    return sizeof(mEps);
}

void RmsnormPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    assert(d == a + getSerializationSize());
}

void RmsnormPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RmsnormPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RmsnormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

RmsnormPluginCreator::RmsnormPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RmsnormPluginCreator::getPluginName() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsnormPluginCreator::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

const PluginFieldCollection* RmsnormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsnormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept  // 通过pluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来，然后调用上文提到的第一个构造函数去创建plugin
{
    const PluginField* fields = fc->fields;
    float eps;
    // bool useDiffOfSquares;
    // nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        // else if (!strcmp(attrName, "use_diff_of_squares"))
        // {
        //     PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
        //     useDiffOfSquares = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        // }
        // else if (!strcmp(attrName, "type_id"))
        // {
        //     PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
        //     type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        // }
    }
    try
    {
        auto* obj = new RmsnormPlugin(eps);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsnormPluginCreator::deserializePlugin(  //这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LayernormPlugin::destroy()
    try
    {
        auto* obj = new RmsnormPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void RmsnormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RmsnormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

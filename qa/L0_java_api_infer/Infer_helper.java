// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import com.google.gson.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.tritonserver.tritonserver.*;
import static org.bytedeco.tritonserver.global.tritonserver.*;

public class InferHelper {
int StringToTritonDataType(String dtype) {
    switch (dtype) { 
        case "BOOL":
        return TRITONSERVER_TYPE_BOOL;
        case "UINT8":
        return TRITONSERVER_TYPE_UINT8;
        case "UINT16":
        return TRITONSERVER_TYPE_UINT16;
        case "UINT32":
        return TRITONSERVER_TYPE_UINT32;
        case "UINT64":
        return TRITONSERVER_TYPE_UINT64;
        case "INT8":
        return TRITONSERVER_TYPE_INT8;
        case "INT16":
        return TRITONSERVER_TYPE_INT16;
        case "INT32":
        return TRITONSERVER_TYPE_INT32;
        case "INT64":
        return TRITONSERVER_TYPE_INT64;
        case "FP16":
        return TRITONSERVER_TYPE_FP16;
        case "FP32":
        return TRITONSERVER_TYPE_FP32;
        case "FP64":
        return TRITONSERVER_TYPE_FP64;
        case "STRING":
        return TRITONSERVER_TYPE_BYTES;
    }
}

static void
GenerateInputData(
    BytePointer[] input_data, int length) {
  input_data[0] = new BytePointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}

static void
GenerateInputData(
    ShortPointer[] input_data, int length) {
  input_data[0] = new ShortPointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}

static void
GenerateInputData(
    IntPointer[] input_data, int length) {
  input_data[0] = new IntPointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}
static void
GenerateInputData(
    LongPointer[] input_data, int length) {
  input_data[0] = new LongPointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}
static void
GenerateInputData(
    FloatPointer[] input_data, int length) {
  input_data[0] = new FloatPointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}

static void
GenerateInputData(
    DoublePointer[] input_data, int length) {
  input_data[0] = new DoublePointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}

static void
GenerateInputData(
    CLongPointer[] input_data, int length) {
  input_data[0] = new CLongPointer(length);
  for (int i = 0; i < length; ++i) {
    input_data[0].put(i, i);
  }
}

};

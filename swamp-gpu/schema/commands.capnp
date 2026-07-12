@0x8c9b1f3a2d4e5f7a;

using TensorId = UInt64;
using KernelId = UInt64;

struct TensorDesc {
    id @0 :TensorId;
    shape @1 :List(UInt64);
    dtype @2 :DType;
    location @3 :MemLocation;
}

enum DType {
    f32 @0;
    f16 @1;
    q4k @2;
    q6k @3;
}

enum MemLocation {
    hostPinned @0;
    deviceLocal @1;
    unifiedMapped @2;
}

struct KernelDispatch {
    kernel @0 :Text;
    inputs @1 :List(TensorId);
    outputs @2 :List(TensorId);
    params @3 :Data;
}

struct BufferCopy {
    srcId @0 :TensorId;
    dstId @1 :TensorId;
    srcOffset @2 :UInt64;
    dstOffset @3 :UInt64;
    size @4 :UInt64;
}

struct Commit {
    frameId @0 :UInt64;
    timestamp @1 :UInt64;
}

union Command {
    tensorCreate @0 :TensorDesc;
    tensorUpload @1 :BufferCopy;
    tensorDownload @2 :BufferCopy;
    tensorFree @3 :TensorId;
    kernelDispatch @4 :KernelDispatch;
    commit @5 :Commit;
    shutdown @6 :Void;
}

struct CommandBatch {
    commands @0 :List(Command);
    batchId @1 :UInt64;
}

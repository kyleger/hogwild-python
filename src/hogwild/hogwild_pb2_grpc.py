# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import hogwild_pb2 as hogwild__pb2


class HogwildStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetNodeInfo = channel.unary_unary(
        '/Hogwild/GetNodeInfo',
        request_serializer=hogwild__pb2.NetworkInfo.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.StartSGD = channel.unary_unary(
        '/Hogwild/StartSGD',
        request_serializer=hogwild__pb2.StartMessage.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.GetWeightUpdate = channel.unary_unary(
        '/Hogwild/GetWeightUpdate',
        request_serializer=hogwild__pb2.WeightUpdate.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.GetReadyToGo = channel.unary_unary(
        '/Hogwild/GetReadyToGo',
        request_serializer=hogwild__pb2.ReadyToGo.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.GetEpochsDone = channel.unary_unary(
        '/Hogwild/GetEpochsDone',
        request_serializer=hogwild__pb2.EpochsDone.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.GetStopMessage = channel.unary_unary(
        '/Hogwild/GetStopMessage',
        request_serializer=hogwild__pb2.StopMessage.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )
    self.GetLossMessage = channel.unary_unary(
        '/Hogwild/GetLossMessage',
        request_serializer=hogwild__pb2.LossMessage.SerializeToString,
        response_deserializer=hogwild__pb2.Empty.FromString,
        )


class HogwildServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetNodeInfo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StartSGD(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetWeightUpdate(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetReadyToGo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetEpochsDone(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetStopMessage(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetLossMessage(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_HogwildServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetNodeInfo': grpc.unary_unary_rpc_method_handler(
          servicer.GetNodeInfo,
          request_deserializer=hogwild__pb2.NetworkInfo.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'StartSGD': grpc.unary_unary_rpc_method_handler(
          servicer.StartSGD,
          request_deserializer=hogwild__pb2.StartMessage.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'GetWeightUpdate': grpc.unary_unary_rpc_method_handler(
          servicer.GetWeightUpdate,
          request_deserializer=hogwild__pb2.WeightUpdate.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'GetReadyToGo': grpc.unary_unary_rpc_method_handler(
          servicer.GetReadyToGo,
          request_deserializer=hogwild__pb2.ReadyToGo.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'GetEpochsDone': grpc.unary_unary_rpc_method_handler(
          servicer.GetEpochsDone,
          request_deserializer=hogwild__pb2.EpochsDone.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'GetStopMessage': grpc.unary_unary_rpc_method_handler(
          servicer.GetStopMessage,
          request_deserializer=hogwild__pb2.StopMessage.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
      'GetLossMessage': grpc.unary_unary_rpc_method_handler(
          servicer.GetLossMessage,
          request_deserializer=hogwild__pb2.LossMessage.FromString,
          response_serializer=hogwild__pb2.Empty.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Hogwild', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))

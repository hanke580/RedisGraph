#include "redismodule.h"

int REDISMODULE_API_FUNC(RedisModule_SubscribeToServerEvent)(RedisModuleCtx *ctx,
															 RedisModuleEvent event, RedisModuleEventCallback callback);

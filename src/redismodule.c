#include "redismodule.h"

void *REDISMODULE_API_FUNC(RedisModule_Alloc)(size_t bytes);
void *REDISMODULE_API_FUNC(RedisModule_Realloc)(void *ptr, size_t bytes);
void REDISMODULE_API_FUNC(RedisModule_Free)(void *ptr);
void *REDISMODULE_API_FUNC(RedisModule_Calloc)(size_t nmemb, size_t size);
char *REDISMODULE_API_FUNC(RedisModule_Strdup)(const char *str);
int REDISMODULE_API_FUNC(RedisModule_GetApi)(const char *, void *);
int REDISMODULE_API_FUNC(RedisModule_CreateCommand)(RedisModuleCtx *ctx, const char *name,
													RedisModuleCmdFunc cmdfunc, const char *strflags, int firstkey, int lastkey, int keystep);
void REDISMODULE_API_FUNC(RedisModule_SetModuleAttribs)(RedisModuleCtx *ctx, const char *name,
														int ver, int apiver);
int REDISMODULE_API_FUNC(RedisModule_IsModuleNameBusy)(const char *name);
int REDISMODULE_API_FUNC(RedisModule_WrongArity)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithLongLong)(RedisModuleCtx *ctx, long long ll);
int REDISMODULE_API_FUNC(RedisModule_GetSelectedDb)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_SelectDb)(RedisModuleCtx *ctx, int newid);
void *REDISMODULE_API_FUNC(RedisModule_OpenKey)(RedisModuleCtx *ctx, RedisModuleString *keyname,
												int mode);
void REDISMODULE_API_FUNC(RedisModule_CloseKey)(RedisModuleKey *kp);
int REDISMODULE_API_FUNC(RedisModule_KeyType)(RedisModuleKey *kp);
size_t REDISMODULE_API_FUNC(RedisModule_ValueLength)(RedisModuleKey *kp);
int REDISMODULE_API_FUNC(RedisModule_ListPush)(RedisModuleKey *kp, int where,
											   RedisModuleString *ele);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_ListPop)(RedisModuleKey *key, int where);
RedisModuleCallReply *REDISMODULE_API_FUNC(RedisModule_Call)(RedisModuleCtx *ctx,
															 const char *cmdname, const char *fmt, ...);
const char *REDISMODULE_API_FUNC(RedisModule_CallReplyProto)(RedisModuleCallReply *reply,
															 size_t *len);
void REDISMODULE_API_FUNC(RedisModule_FreeCallReply)(RedisModuleCallReply *reply);
int REDISMODULE_API_FUNC(RedisModule_CallReplyType)(RedisModuleCallReply *reply);
long long REDISMODULE_API_FUNC(RedisModule_CallReplyInteger)(RedisModuleCallReply *reply);
size_t REDISMODULE_API_FUNC(RedisModule_CallReplyLength)(RedisModuleCallReply *reply);
RedisModuleCallReply *REDISMODULE_API_FUNC(RedisModule_CallReplyArrayElement)(
	RedisModuleCallReply *reply, size_t idx);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateString)(RedisModuleCtx *ctx,
																  const char *ptr, size_t len);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringFromLongLong)(RedisModuleCtx *ctx,
																			  long long ll);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringFromDouble)(RedisModuleCtx *ctx,
																			double d);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringFromLongDouble)(RedisModuleCtx *ctx,
																				long double ld, int humanfriendly);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringFromString)(RedisModuleCtx *ctx,
																			const RedisModuleString *str);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringPrintf)(RedisModuleCtx *ctx,
																		const char *fmt, ...);
void REDISMODULE_API_FUNC(RedisModule_FreeString)(RedisModuleCtx *ctx, RedisModuleString *str);
const char *REDISMODULE_API_FUNC(RedisModule_StringPtrLen)(const RedisModuleString *str,
														   size_t *len);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithError)(RedisModuleCtx *ctx, const char *err);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithSimpleString)(RedisModuleCtx *ctx, const char *msg);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithArray)(RedisModuleCtx *ctx, long len);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithNullArray)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithEmptyArray)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_ReplySetArrayLength)(RedisModuleCtx *ctx, long len);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithStringBuffer)(RedisModuleCtx *ctx, const char *buf,
															size_t len);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithCString)(RedisModuleCtx *ctx, const char *buf);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithString)(RedisModuleCtx *ctx, RedisModuleString *str);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithEmptyString)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithVerbatimString)(RedisModuleCtx *ctx, const char *buf,
															  size_t len);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithNull)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithDouble)(RedisModuleCtx *ctx, double d);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithLongDouble)(RedisModuleCtx *ctx, long double d);
int REDISMODULE_API_FUNC(RedisModule_ReplyWithCallReply)(RedisModuleCtx *ctx,
														 RedisModuleCallReply *reply);
int REDISMODULE_API_FUNC(RedisModule_StringToLongLong)(const RedisModuleString *str, long long *ll);
int REDISMODULE_API_FUNC(RedisModule_StringToDouble)(const RedisModuleString *str, double *d);
int REDISMODULE_API_FUNC(RedisModule_StringToLongDouble)(const RedisModuleString *str,
														 long double *d);
void REDISMODULE_API_FUNC(RedisModule_AutoMemory)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_Replicate)(RedisModuleCtx *ctx, const char *cmdname,
												const char *fmt, ...);
int REDISMODULE_API_FUNC(RedisModule_ReplicateVerbatim)(RedisModuleCtx *ctx);
const char *REDISMODULE_API_FUNC(RedisModule_CallReplyStringPtr)(RedisModuleCallReply *reply,
																 size_t *len);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CreateStringFromCallReply)(
	RedisModuleCallReply *reply);
int REDISMODULE_API_FUNC(RedisModule_DeleteKey)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_UnlinkKey)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_StringSet)(RedisModuleKey *key, RedisModuleString *str);
char *REDISMODULE_API_FUNC(RedisModule_StringDMA)(RedisModuleKey *key, size_t *len, int mode);
int REDISMODULE_API_FUNC(RedisModule_StringTruncate)(RedisModuleKey *key, size_t newlen);
mstime_t REDISMODULE_API_FUNC(RedisModule_GetExpire)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_SetExpire)(RedisModuleKey *key, mstime_t expire);
void REDISMODULE_API_FUNC(RedisModule_ResetDataset)(int restart_aof, int async);
unsigned long long REDISMODULE_API_FUNC(RedisModule_DbSize)(RedisModuleCtx *ctx);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_RandomKey)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_ZsetAdd)(RedisModuleKey *key, double score,
											  RedisModuleString *ele, int *flagsptr);
int REDISMODULE_API_FUNC(RedisModule_ZsetIncrby)(RedisModuleKey *key, double score,
												 RedisModuleString *ele, int *flagsptr, double *newscore);
int REDISMODULE_API_FUNC(RedisModule_ZsetScore)(RedisModuleKey *key, RedisModuleString *ele,
												double *score);
int REDISMODULE_API_FUNC(RedisModule_ZsetRem)(RedisModuleKey *key, RedisModuleString *ele,
											  int *deleted);
void REDISMODULE_API_FUNC(RedisModule_ZsetRangeStop)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_ZsetFirstInScoreRange)(RedisModuleKey *key, double min,
															double max, int minex, int maxex);
int REDISMODULE_API_FUNC(RedisModule_ZsetLastInScoreRange)(RedisModuleKey *key, double min,
														   double max, int minex, int maxex);
int REDISMODULE_API_FUNC(RedisModule_ZsetFirstInLexRange)(RedisModuleKey *key,
														  RedisModuleString *min, RedisModuleString *max);
int REDISMODULE_API_FUNC(RedisModule_ZsetLastInLexRange)(RedisModuleKey *key,
														 RedisModuleString *min, RedisModuleString *max);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_ZsetRangeCurrentElement)(RedisModuleKey *key,
																			 double *score);
int REDISMODULE_API_FUNC(RedisModule_ZsetRangeNext)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_ZsetRangePrev)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_ZsetRangeEndReached)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_HashSet)(RedisModuleKey *key, int flags, ...);
int REDISMODULE_API_FUNC(RedisModule_HashGet)(RedisModuleKey *key, int flags, ...);
int REDISMODULE_API_FUNC(RedisModule_IsKeysPositionRequest)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_KeyAtPos)(RedisModuleCtx *ctx, int pos);
unsigned long long REDISMODULE_API_FUNC(RedisModule_GetClientId)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_GetClientInfoById)(void *ci, uint64_t id);
int REDISMODULE_API_FUNC(RedisModule_PublishMessage)(RedisModuleCtx *ctx,
													 RedisModuleString *channel, RedisModuleString *message);
int REDISMODULE_API_FUNC(RedisModule_GetContextFlags)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_AvoidReplicaTraffic)();
void *REDISMODULE_API_FUNC(RedisModule_PoolAlloc)(RedisModuleCtx *ctx, size_t bytes);
RedisModuleType *REDISMODULE_API_FUNC(RedisModule_CreateDataType)(RedisModuleCtx *ctx,
																  const char *name, int encver, RedisModuleTypeMethods *typemethods);
int REDISMODULE_API_FUNC(RedisModule_ModuleTypeSetValue)(RedisModuleKey *key, RedisModuleType *mt,
														 void *value);
int REDISMODULE_API_FUNC(RedisModule_ModuleTypeReplaceValue)(RedisModuleKey *key,
															 RedisModuleType *mt, void *new_value, void **old_value);
RedisModuleType *REDISMODULE_API_FUNC(RedisModule_ModuleTypeGetType)(RedisModuleKey *key);
void *REDISMODULE_API_FUNC(RedisModule_ModuleTypeGetValue)(RedisModuleKey *key);
int REDISMODULE_API_FUNC(RedisModule_IsIOError)(RedisModuleIO *io);
void REDISMODULE_API_FUNC(RedisModule_SetModuleOptions)(RedisModuleCtx *ctx, int options);
int REDISMODULE_API_FUNC(RedisModule_SignalModifiedKey)(RedisModuleCtx *ctx,
														RedisModuleString *keyname);
void REDISMODULE_API_FUNC(RedisModule_SaveUnsigned)(RedisModuleIO *io, uint64_t value);
uint64_t REDISMODULE_API_FUNC(RedisModule_LoadUnsigned)(RedisModuleIO *io);
void REDISMODULE_API_FUNC(RedisModule_SaveSigned)(RedisModuleIO *io, int64_t value);
int64_t REDISMODULE_API_FUNC(RedisModule_LoadSigned)(RedisModuleIO *io);
void REDISMODULE_API_FUNC(RedisModule_EmitAOF)(RedisModuleIO *io, const char *cmdname,
											   const char *fmt, ...);
void REDISMODULE_API_FUNC(RedisModule_SaveString)(RedisModuleIO *io, RedisModuleString *s);
void REDISMODULE_API_FUNC(RedisModule_SaveStringBuffer)(RedisModuleIO *io, const char *str,
														size_t len);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_LoadString)(RedisModuleIO *io);
char *REDISMODULE_API_FUNC(RedisModule_LoadStringBuffer)(RedisModuleIO *io, size_t *lenptr);
void REDISMODULE_API_FUNC(RedisModule_SaveDouble)(RedisModuleIO *io, double value);
double REDISMODULE_API_FUNC(RedisModule_LoadDouble)(RedisModuleIO *io);
void REDISMODULE_API_FUNC(RedisModule_SaveFloat)(RedisModuleIO *io, float value);
float REDISMODULE_API_FUNC(RedisModule_LoadFloat)(RedisModuleIO *io);
void REDISMODULE_API_FUNC(RedisModule_SaveLongDouble)(RedisModuleIO *io, long double value);
long double REDISMODULE_API_FUNC(RedisModule_LoadLongDouble)(RedisModuleIO *io);
void *REDISMODULE_API_FUNC(RedisModule_LoadDataTypeFromString)(const RedisModuleString *str,
															   const RedisModuleType *mt);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_SaveDataTypeToString)(RedisModuleCtx *ctx,
																		  void *data, const RedisModuleType *mt);
void REDISMODULE_API_FUNC(RedisModule_Log)(RedisModuleCtx *ctx, const char *level, const char *fmt,
										   ...);
void REDISMODULE_API_FUNC(RedisModule_LogIOError)(RedisModuleIO *io, const char *levelstr,
												  const char *fmt, ...);
void REDISMODULE_API_FUNC(RedisModule__Assert)(const char *estr, const char *file, int line);
void REDISMODULE_API_FUNC(RedisModule_LatencyAddSample)(const char *event, mstime_t latency);
int REDISMODULE_API_FUNC(RedisModule_StringAppendBuffer)(RedisModuleCtx *ctx,
														 RedisModuleString *str, const char *buf, size_t len);
void REDISMODULE_API_FUNC(RedisModule_RetainString)(RedisModuleCtx *ctx, RedisModuleString *str);
int REDISMODULE_API_FUNC(RedisModule_StringCompare)(RedisModuleString *a, RedisModuleString *b);
RedisModuleCtx *REDISMODULE_API_FUNC(RedisModule_GetContextFromIO)(RedisModuleIO *io);
const RedisModuleString *REDISMODULE_API_FUNC(RedisModule_GetKeyNameFromIO)(RedisModuleIO *io);
const RedisModuleString *REDISMODULE_API_FUNC(RedisModule_GetKeyNameFromModuleKey)(
	RedisModuleKey *key);
long long REDISMODULE_API_FUNC(RedisModule_Milliseconds)(void);
void REDISMODULE_API_FUNC(RedisModule_DigestAddStringBuffer)(RedisModuleDigest *md,
															 unsigned char *ele, size_t len);
void REDISMODULE_API_FUNC(RedisModule_DigestAddLongLong)(RedisModuleDigest *md, long long ele);
void REDISMODULE_API_FUNC(RedisModule_DigestEndSequence)(RedisModuleDigest *md);
RedisModuleDict *REDISMODULE_API_FUNC(RedisModule_CreateDict)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_FreeDict)(RedisModuleCtx *ctx, RedisModuleDict *d);
uint64_t REDISMODULE_API_FUNC(RedisModule_DictSize)(RedisModuleDict *d);
int REDISMODULE_API_FUNC(RedisModule_DictSetC)(RedisModuleDict *d, void *key, size_t keylen,
											   void *ptr);
int REDISMODULE_API_FUNC(RedisModule_DictReplaceC)(RedisModuleDict *d, void *key, size_t keylen,
												   void *ptr);
int REDISMODULE_API_FUNC(RedisModule_DictSet)(RedisModuleDict *d, RedisModuleString *key,
											  void *ptr);
int REDISMODULE_API_FUNC(RedisModule_DictReplace)(RedisModuleDict *d, RedisModuleString *key,
												  void *ptr);
void *REDISMODULE_API_FUNC(RedisModule_DictGetC)(RedisModuleDict *d, void *key, size_t keylen,
												 int *nokey);
void *REDISMODULE_API_FUNC(RedisModule_DictGet)(RedisModuleDict *d, RedisModuleString *key,
												int *nokey);
int REDISMODULE_API_FUNC(RedisModule_DictDelC)(RedisModuleDict *d, void *key, size_t keylen,
											   void *oldval);
int REDISMODULE_API_FUNC(RedisModule_DictDel)(RedisModuleDict *d, RedisModuleString *key,
											  void *oldval);
RedisModuleDictIter *REDISMODULE_API_FUNC(RedisModule_DictIteratorStartC)(RedisModuleDict *d,
																		  const char *op, void *key, size_t keylen);
RedisModuleDictIter *REDISMODULE_API_FUNC(RedisModule_DictIteratorStart)(RedisModuleDict *d,
																		 const char *op, RedisModuleString *key);
void REDISMODULE_API_FUNC(RedisModule_DictIteratorStop)(RedisModuleDictIter *di);
int REDISMODULE_API_FUNC(RedisModule_DictIteratorReseekC)(RedisModuleDictIter *di, const char *op,
														  void *key, size_t keylen);
int REDISMODULE_API_FUNC(RedisModule_DictIteratorReseek)(RedisModuleDictIter *di, const char *op,
														 RedisModuleString *key);
void *REDISMODULE_API_FUNC(RedisModule_DictNextC)(RedisModuleDictIter *di, size_t *keylen,
												  void **dataptr);
void *REDISMODULE_API_FUNC(RedisModule_DictPrevC)(RedisModuleDictIter *di, size_t *keylen,
												  void **dataptr);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_DictNext)(RedisModuleCtx *ctx,
															  RedisModuleDictIter *di, void **dataptr);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_DictPrev)(RedisModuleCtx *ctx,
															  RedisModuleDictIter *di, void **dataptr);
int REDISMODULE_API_FUNC(RedisModule_DictCompareC)(RedisModuleDictIter *di, const char *op,
												   void *key, size_t keylen);
int REDISMODULE_API_FUNC(RedisModule_DictCompare)(RedisModuleDictIter *di, const char *op,
												  RedisModuleString *key);
int REDISMODULE_API_FUNC(RedisModule_RegisterInfoFunc)(RedisModuleCtx *ctx, RedisModuleInfoFunc cb);
int REDISMODULE_API_FUNC(RedisModule_InfoAddSection)(RedisModuleInfoCtx *ctx, char *name);
int REDISMODULE_API_FUNC(RedisModule_InfoBeginDictField)(RedisModuleInfoCtx *ctx, char *name);
int REDISMODULE_API_FUNC(RedisModule_InfoEndDictField)(RedisModuleInfoCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_InfoAddFieldString)(RedisModuleInfoCtx *ctx, char *field,
														 RedisModuleString *value);
int REDISMODULE_API_FUNC(RedisModule_InfoAddFieldCString)(RedisModuleInfoCtx *ctx, char *field,
														  char *value);
int REDISMODULE_API_FUNC(RedisModule_InfoAddFieldDouble)(RedisModuleInfoCtx *ctx, char *field,
														 double value);
int REDISMODULE_API_FUNC(RedisModule_InfoAddFieldLongLong)(RedisModuleInfoCtx *ctx, char *field,
														   long long value);
int REDISMODULE_API_FUNC(RedisModule_InfoAddFieldULongLong)(RedisModuleInfoCtx *ctx, char *field,
															unsigned long long value);
RedisModuleServerInfoData *REDISMODULE_API_FUNC(RedisModule_GetServerInfo)(RedisModuleCtx *ctx,
																		   const char *section);
void REDISMODULE_API_FUNC(RedisModule_FreeServerInfo)(RedisModuleCtx *ctx,
													  RedisModuleServerInfoData *data);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_ServerInfoGetField)(RedisModuleCtx *ctx,
																		RedisModuleServerInfoData *data, const char *field);
const char *REDISMODULE_API_FUNC(RedisModule_ServerInfoGetFieldC)(RedisModuleServerInfoData *data,
																  const char *field);
long long REDISMODULE_API_FUNC(RedisModule_ServerInfoGetFieldSigned)(RedisModuleServerInfoData
																	 *data, const char *field, int *out_err);
unsigned long long REDISMODULE_API_FUNC(RedisModule_ServerInfoGetFieldUnsigned)(
	RedisModuleServerInfoData *data, const char *field, int *out_err);
double REDISMODULE_API_FUNC(RedisModule_ServerInfoGetFieldDouble)(RedisModuleServerInfoData *data,
																  const char *field, int *out_err);
int REDISMODULE_API_FUNC(RedisModule_SubscribeToServerEvent)(RedisModuleCtx *ctx,
															 RedisModuleEvent event, RedisModuleEventCallback callback);
int REDISMODULE_API_FUNC(RedisModule_SetLRU)(RedisModuleKey *key, mstime_t lru_idle);
int REDISMODULE_API_FUNC(RedisModule_GetLRU)(RedisModuleKey *key, mstime_t *lru_idle);
int REDISMODULE_API_FUNC(RedisModule_SetLFU)(RedisModuleKey *key, long long lfu_freq);
int REDISMODULE_API_FUNC(RedisModule_GetLFU)(RedisModuleKey *key, long long *lfu_freq);
RedisModuleBlockedClient *REDISMODULE_API_FUNC(RedisModule_BlockClientOnKeys)(RedisModuleCtx *ctx,
																			  RedisModuleCmdFunc reply_callback, RedisModuleCmdFunc timeout_callback,
																			  void (*free_privdata)(RedisModuleCtx *, void *), long long timeout_ms, RedisModuleString **keys,
																			  int numkeys, void *privdata);
void REDISMODULE_API_FUNC(RedisModule_SignalKeyAsReady)(RedisModuleCtx *ctx,
														RedisModuleString *key);
RedisModuleString *REDISMODULE_API_FUNC(RedisModule_GetBlockedClientReadyKey)(RedisModuleCtx *ctx);
RedisModuleScanCursor *REDISMODULE_API_FUNC(RedisModule_ScanCursorCreate)();
void REDISMODULE_API_FUNC(RedisModule_ScanCursorRestart)(RedisModuleScanCursor *cursor);
void REDISMODULE_API_FUNC(RedisModule_ScanCursorDestroy)(RedisModuleScanCursor *cursor);
int REDISMODULE_API_FUNC(RedisModule_Scan)(RedisModuleCtx *ctx, RedisModuleScanCursor *cursor,
										   RedisModuleScanCB fn, void *privdata);
int REDISMODULE_API_FUNC(RedisModule_ScanKey)(RedisModuleKey *key, RedisModuleScanCursor *cursor,
											  RedisModuleScanKeyCB fn, void *privdata);
/* Experimental APIs */
RedisModuleBlockedClient *REDISMODULE_API_FUNC(RedisModule_BlockClient)(RedisModuleCtx *ctx,
																		RedisModuleCmdFunc reply_callback, RedisModuleCmdFunc timeout_callback,
																		void (*free_privdata)(RedisModuleCtx *, void *), long long timeout_ms);
int REDISMODULE_API_FUNC(RedisModule_UnblockClient)(RedisModuleBlockedClient *bc, void *privdata);
int REDISMODULE_API_FUNC(RedisModule_IsBlockedReplyRequest)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_IsBlockedTimeoutRequest)(RedisModuleCtx *ctx);
void *REDISMODULE_API_FUNC(RedisModule_GetBlockedClientPrivateData)(RedisModuleCtx *ctx);
RedisModuleBlockedClient *REDISMODULE_API_FUNC(RedisModule_GetBlockedClientHandle)(
	RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_AbortBlock)(RedisModuleBlockedClient *bc);
RedisModuleCtx *REDISMODULE_API_FUNC(RedisModule_GetThreadSafeContext)(
	RedisModuleBlockedClient *bc);
void REDISMODULE_API_FUNC(RedisModule_FreeThreadSafeContext)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_ThreadSafeContextLock)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_ThreadSafeContextUnlock)(RedisModuleCtx *ctx);
int REDISMODULE_API_FUNC(RedisModule_SubscribeToKeyspaceEvents)(RedisModuleCtx *ctx, int types,
																RedisModuleNotificationFunc cb);
int REDISMODULE_API_FUNC(RedisModule_NotifyKeyspaceEvent)(RedisModuleCtx *ctx, int type,
														  const char *event, RedisModuleString *key);
int REDISMODULE_API_FUNC(RedisModule_GetNotifyKeyspaceEvents)();
int REDISMODULE_API_FUNC(RedisModule_BlockedClientDisconnected)(RedisModuleCtx *ctx);
void REDISMODULE_API_FUNC(RedisModule_RegisterClusterMessageReceiver)(RedisModuleCtx *ctx,
																	  uint8_t type, RedisModuleClusterMessageReceiver callback);
int REDISMODULE_API_FUNC(RedisModule_SendClusterMessage)(RedisModuleCtx *ctx, char *target_id,
														 uint8_t type, unsigned char *msg, uint32_t len);
int REDISMODULE_API_FUNC(RedisModule_GetClusterNodeInfo)(RedisModuleCtx *ctx, const char *id,
														 char *ip, char *master_id, int *port, int *flags);
char **REDISMODULE_API_FUNC(RedisModule_GetClusterNodesList)(RedisModuleCtx *ctx, size_t *numnodes);
void REDISMODULE_API_FUNC(RedisModule_FreeClusterNodesList)(char **ids);
RedisModuleTimerID REDISMODULE_API_FUNC(RedisModule_CreateTimer)(RedisModuleCtx *ctx,
																 mstime_t period, RedisModuleTimerProc callback, void *data);
int REDISMODULE_API_FUNC(RedisModule_StopTimer)(RedisModuleCtx *ctx, RedisModuleTimerID id,
												void **data);
int REDISMODULE_API_FUNC(RedisModule_GetTimerInfo)(RedisModuleCtx *ctx, RedisModuleTimerID id,
												   uint64_t *remaining, void **data);
const char *REDISMODULE_API_FUNC(RedisModule_GetMyClusterID)(void);
size_t REDISMODULE_API_FUNC(RedisModule_GetClusterSize)(void);
void REDISMODULE_API_FUNC(RedisModule_GetRandomBytes)(unsigned char *dst, size_t len);
void REDISMODULE_API_FUNC(RedisModule_GetRandomHexChars)(char *dst, size_t len);
void REDISMODULE_API_FUNC(RedisModule_SetDisconnectCallback)(RedisModuleBlockedClient *bc,
															 RedisModuleDisconnectFunc callback);
void REDISMODULE_API_FUNC(RedisModule_SetClusterFlags)(RedisModuleCtx *ctx, uint64_t flags);
int REDISMODULE_API_FUNC(RedisModule_ExportSharedAPI)(RedisModuleCtx *ctx, const char *apiname,
													  void *func);
void *REDISMODULE_API_FUNC(RedisModule_GetSharedAPI)(RedisModuleCtx *ctx, const char *apiname);
RedisModuleCommandFilter *REDISMODULE_API_FUNC(RedisModule_RegisterCommandFilter)(
	RedisModuleCtx *ctx, RedisModuleCommandFilterFunc cb, int flags);
int REDISMODULE_API_FUNC(RedisModule_UnregisterCommandFilter)(RedisModuleCtx *ctx,
															  RedisModuleCommandFilter *filter);
int REDISMODULE_API_FUNC(RedisModule_CommandFilterArgsCount)(RedisModuleCommandFilterCtx *fctx);
const RedisModuleString *REDISMODULE_API_FUNC(RedisModule_CommandFilterArgGet)(
	RedisModuleCommandFilterCtx *fctx, int pos);
int REDISMODULE_API_FUNC(RedisModule_CommandFilterArgInsert)(RedisModuleCommandFilterCtx *fctx,
															 int pos, RedisModuleString *arg);
int REDISMODULE_API_FUNC(RedisModule_CommandFilterArgReplace)(RedisModuleCommandFilterCtx *fctx,
															  int pos, RedisModuleString *arg);
int REDISMODULE_API_FUNC(RedisModule_CommandFilterArgDelete)(RedisModuleCommandFilterCtx *fctx,
															 int pos);
int REDISMODULE_API_FUNC(RedisModule_Fork)(RedisModuleForkDoneHandler cb, void *user_data);
int REDISMODULE_API_FUNC(RedisModule_ExitFromChild)(int retcode);
int REDISMODULE_API_FUNC(RedisModule_KillForkChild)(int child_pid);
float REDISMODULE_API_FUNC(RedisModule_GetUsedMemoryRatio)();
size_t REDISMODULE_API_FUNC(RedisModule_MallocSize)(void *ptr);
RedisModuleUser *REDISMODULE_API_FUNC(RedisModule_CreateModuleUser)(const char *name);
void REDISMODULE_API_FUNC(RedisModule_FreeModuleUser)(RedisModuleUser *user);
int REDISMODULE_API_FUNC(RedisModule_SetModuleUserACL)(RedisModuleUser *user, const char *acl);
int REDISMODULE_API_FUNC(RedisModule_AuthenticateClientWithACLUser)(RedisModuleCtx *ctx,
																	const char *name, size_t len, RedisModuleUserChangedFunc callback, void *privdata,
																	uint64_t *client_id);
int REDISMODULE_API_FUNC(RedisModule_AuthenticateClientWithUser)(RedisModuleCtx *ctx,
																 RedisModuleUser *user, RedisModuleUserChangedFunc callback, void *privdata, uint64_t *client_id);
void REDISMODULE_API_FUNC(RedisModule_DeauthenticateAndCloseClient)(RedisModuleCtx *ctx,
																	uint64_t client_id);

/*
* Copyright 2018-2020 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "execution_plan_clone.h"
#include "../RG.h"
#include "../query_ctx.h"
#include "../util/rax_extensions.h"
#include "execution_plan_build/execution_plan_modify.h"



static ExecutionPlan *_ClonePlanInternals(const ExecutionPlan *template_) {
	ExecutionPlan *clone = ExecutionPlan_NewEmptyExecutionPlan();

	clone->record_map = raxClone(template_->record_map);
	if(template_->ast_segment) clone->ast_segment = AST_ShallowCopy(template_->ast_segment);
	if(template_->query_graph) {
		QueryGraph_ResolveUnknownRelIDs(template_->query_graph);
		clone->query_graph = QueryGraph_Clone(template_->query_graph);
	}
	// TODO improve QueryGraph logic so that we do not need to store or clone connected_components.
	if(template_->connected_components) {
		array_clone_with_cb(clone->connected_components, template_->connected_components, QueryGraph_Clone);
	}

	// Temporarily set the thread-local AST to be the one referenced by this ExecutionPlan segment.
	QueryCtx_SetAST(clone->ast_segment);

	return clone;
}



static OpBase *_CloneOpTree(OpBase *template_parent, OpBase *template_current,
							OpBase *clone_parent) {
	const ExecutionPlan *plan_segment;
	if(!template_parent || (template_current->plan != template_parent->plan)) {
		/* If this is the first operation or it was built using a different ExecutionPlan
		 * segment than its parent, clone the ExecutionPlan segment. */
		plan_segment = _ClonePlanInternals(template_current->plan);
	} else {
		// This op was built as part of the same segment as its parent, don't change ExecutionPlans.
		plan_segment = clone_parent->plan;
	}

	// Clone the current operation.
	OpBase *clone_current = OpBase_Clone(plan_segment, template_current);

	for(uint i = 0; i < template_current->childCount; i++) {
		// Recursively visit and clone the op's children.
		OpBase *child_op = _CloneOpTree(template_current, template_current->children[i], clone_current);
		ExecutionPlan_AddOp(clone_current, child_op);
	}

	return clone_current;
}

static ExecutionPlan *_ExecutionPlan_Clone(const ExecutionPlan *template_) {
	OpBase *clone_root = _CloneOpTree(NULL, template_->root, NULL);
	// The "master" execution plan is the one constructed with the root op.
	ExecutionPlan *clone = (ExecutionPlan *)clone_root->plan;
	// The root op is currently NULL; set it now.
	clone->root = clone_root;

	return clone;
}

/* This function clones the input ExecutionPlan by recursively visiting its tree of ops.
 * When an op is encountered that was constructed as part of a different ExecutionPlan segment, that segment
 * and its internal members (FilterTree, record mapping, query graphs, and AST segment) are also cloned. */
ExecutionPlan *ExecutionPlan_Clone(const ExecutionPlan *template_) {
	ASSERT(template_ != NULL);
	// Store the original AST pointer.
	AST *master_ast = QueryCtx_GetAST();
	// Verify that the execution plan template is not prepared yet.
	ASSERT(template_->prepared == false && "Execution plan cloning should be only on templates");
	ExecutionPlan *clone = _ExecutionPlan_Clone(template_);
	// Restore the original AST pointer.
	QueryCtx_SetAST(master_ast);
	return clone;
}


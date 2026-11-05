/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/ob_resolver.h"
#include "lib/utility/ob_smart_call.h"
#include "lib/stat/ob_diagnostic_info_guard.h"
#include "sql/resolver/cmd/ob_alter_system_resolver.h"
#include "sql/resolver/dml/ob_insert_resolver.h"
#include "sql/resolver/dml/ob_update_resolver.h"
#include "sql/resolver/dml/ob_delete_resolver.h"
#include "sql/resolver/tcl/ob_start_trans_resolver.h"
#include "sql/resolver/tcl/ob_end_trans_resolver.h"
#include "sql/resolver/prepare/ob_prepare_resolver.h"
#include "sql/resolver/prepare/ob_execute_resolver.h"
#include "sql/resolver/prepare/ob_deallocate_resolver.h"
#include "sql/resolver/ddl/ob_create_table_resolver.h"
#include "sql/resolver/ddl/ob_rename_table_resolver.h"
#include "sql/resolver/ddl/ob_truncate_table_resolver.h"
#include "sql/resolver/ddl/ob_fork_table_resolver.h"
#include "sql/resolver/ddl/ob_fork_database_resolver.h"
#include "sql/resolver/ddl/ob_create_table_like_resolver.h"
#include "sql/resolver/ddl/ob_alter_table_resolver.h"
#include "sql/resolver/ddl/ob_drop_table_resolver.h"
#include "sql/resolver/ddl/ob_create_index_resolver.h"
#include "sql/resolver/ddl/ob_drop_index_resolver.h"
#include "sql/resolver/ddl/ob_create_database_resolver.h"
#include "sql/resolver/ddl/ob_alter_database_resolver.h"
#include "sql/resolver/ddl/ob_use_database_resolver.h"
#include "sql/resolver/ddl/ob_drop_database_resolver.h"
#include "sql/resolver/ddl/ob_create_view_resolver.h"
#include "sql/resolver/ddl/ob_explain_resolver.h"
#include "sql/resolver/ddl/ob_create_outline_resolver.h"
#include "sql/resolver/ddl/ob_alter_outline_resolver.h"
#include "sql/resolver/ddl/ob_drop_outline_resolver.h"
#include "sql/resolver/ddl/ob_drop_routine_resolver.h"
#include "sql/resolver/ddl/ob_trigger_resolver.h"
#include "sql/resolver/ddl/ob_optimize_resolver.h"
#include "ddl/ob_drop_routine_resolver.h"
#include "ddl/ob_alter_routine_resolver.h"
#include "sql/resolver/ddl/ob_create_package_resolver.h"
#include "sql/resolver/ddl/ob_drop_package_resolver.h"
#include "sql/resolver/ddl/ob_recyclebin_restore_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_analyze_stmt_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_set_comment_resolver.h"
#include "sql/resolver/ddl/ob_lock_table_resolver.h"
#include "sql/resolver/dml/ob_insert_resolver.h"
#include "sql/resolver/dml/ob_update_resolver.h"
#include "sql/resolver/dml/ob_delete_resolver.h"
#include "sql/resolver/dcl/ob_create_user_resolver.h"
#include "sql/resolver/dcl/ob_drop_user_resolver.h"
#include "sql/resolver/dcl/ob_rename_user_resolver.h"
#include "sql/resolver/dcl/ob_set_password_resolver.h"
#include "sql/resolver/dcl/ob_lock_user_resolver.h"
#include "sql/resolver/dcl/ob_grant_resolver.h"
#include "sql/resolver/dcl/ob_revoke_resolver.h"
#include "sql/resolver/dcl/ob_create_role_resolver.h"
#include "sql/resolver/dcl/ob_drop_role_resolver.h"
#include "sql/resolver/dcl/ob_alter_user_role_resolver.h"
#include "sql/resolver/tcl/ob_start_trans_resolver.h"
#include "sql/resolver/tcl/ob_end_trans_resolver.h"
#include "tcl/ob_savepoint_resolver.h"
#include "sql/resolver/cmd/ob_variable_set_resolver.h"
#include "sql/resolver/cmd/ob_show_resolver.h"
#include "sql/resolver/cmd/ob_diff_table_resolver.h"
#include "sql/resolver/cmd/ob_merge_table_resolver.h"
#include "sql/resolver/cmd/ob_kill_resolver.h"
#include "sql/resolver/cmd/ob_set_names_resolver.h"
#include "sql/resolver/cmd/ob_set_transaction_resolver.h"
#include "sql/resolver/cmd/ob_empty_query_resolver.h"
#include "sql/resolver/cmd/ob_anonymous_block_resolver.h"
#include "sql/resolver/cmd/ob_call_procedure_resolver.h"
#include "sql/resolver/cmd/ob_load_data_resolver.h"
#include "sql/resolver/prepare/ob_execute_resolver.h"
#include "sql/resolver/prepare/ob_deallocate_resolver.h"
#include "sql/resolver/ddl/ob_purge_resolver.h"
#include "sql/resolver/ddl/ob_set_comment_resolver.h"
#include "sql/resolver/expr/ob_raw_expr_wrap_enum_set.h"
#include "sql/resolver/cmd/ob_get_diagnostics_resolver.h"
#include "sql/resolver/dcl/ob_alter_role_resolver.h"
#include "sql/pl/ob_pl_package.h"

namespace oceanbase
{
using namespace common;
using namespace share;
namespace sql
{
ObResolver::ObResolver(ObResolverParams &params)
    : params_(params)
{

}

ObResolver::~ObResolver()
{
}

template <typename ResolverType>
int ObResolver::stmt_resolver_func(ObResolverParams &params, const ParseNode &parse_tree, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  HEAP_VAR(ResolverType, stmt_resolver, params) {
    if (OB_FAIL(SMART_CALL(stmt_resolver.resolve(parse_tree)))) {
    }
    stmt = stmt_resolver.get_basic_stmt();
  }
  return ret;
}

template <typename SelectResolverType>
int ObResolver::select_stmt_resolver_func(ObResolverParams &params, const ParseNode &parse_tree, ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  HEAP_VAR(SelectResolverType, stmt_resolver, params) {
    stmt_resolver.set_calc_found_rows(true);
    stmt_resolver.set_has_top_limit(true);
    if (OB_FAIL(SMART_CALL(stmt_resolver.resolve(parse_tree)))) {
    }
    stmt = stmt_resolver.get_basic_stmt();
  }
  return ret;
}


static const char *describe_type_name(const ObObjType type)
{
  const char *name = "VARCHAR";
  switch (type) {
    case ObTinyIntType:
      name = "BOOLEAN";
      break;
    case ObUTinyIntType:
    case ObSmallIntType:
    case ObUSmallIntType:
    case ObMediumIntType:
    case ObUMediumIntType:
    case ObInt32Type:
    case ObUInt32Type:
    case ObIntType:
    case ObUInt64Type:
    case ObBitType:
      name = "BIGINT";
      break;
    case ObFloatType:
    case ObUFloatType:
    case ObDoubleType:
    case ObUDoubleType:
    case ObNumberType:
    case ObUNumberType:
      name = "DOUBLE";
      break;
    case ObDateType:
      name = "DATE";
      break;
    case ObDateTimeType:
    case ObTimestampType:
      name = "DATETIME";
      break;
    case ObVarcharType:
    case ObCharType:
    case ObTextType:
    case ObTinyTextType:
    case ObMediumTextType:
    case ObLongTextType:
      name = "VARCHAR";
      break;
    default:
      name = "VARCHAR";
      break;
  }
  return name;
}

int ObResolver::resolve_describe_select(ObResolverParams &params,
                                        const ParseNode &parse_tree,
                                        ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  LOG_INFO("describe select resolver entered");
  ObStmt *child_stmt = NULL;
  ObSelectStmt *sel = NULL;
  const ParseNode *child = parse_tree.num_child_ > 0 ? parse_tree.children_[0] : NULL;
  ObSqlString sql;
  if (OB_ISNULL(child) || OB_ISNULL(params.allocator_) || OB_ISNULL(params.session_info_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (T_SELECT != child->type_) {
    ret = OB_NOT_SUPPORTED;
    LOG_INFO("describe child type mismatch", "child_type", static_cast<int64_t>(child->type_),
             "num_child", child->num_child_);
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "describe a non-select statement");
  } else {
    ObResolver inner(params);
    int inner_ret = inner.resolve(ObResolver::IS_NOT_PREPARED_STMT, *child, child_stmt);
    LOG_INFO("describe inner resolve", "ret", inner_ret, "child_type", static_cast<int64_t>(child->type_),
             "stmt_null", (NULL == child_stmt));
    if (OB_FAIL(inner_ret)) {
      ret = inner_ret;
    } else if (OB_ISNULL(child_stmt) || !child_stmt->is_select_stmt()) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      sel = static_cast<ObSelectStmt *>(child_stmt);
    }
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < sel->get_select_item_size(); ++i) {
    const SelectItem &item = sel->get_select_item(i);
    ObString col_name = item.alias_name_.empty() ? item.expr_name_ : item.alias_name_;
    const char *type_name = "VARCHAR";
    bool nullable = true;
    if (OB_NOT_NULL(item.expr_)) {
      type_name = describe_type_name(item.expr_->get_result_type().get_type());
      if (T_REF_COLUMN == item.expr_->get_expr_type()) {
        ObColumnRefRawExpr *col = static_cast<ObColumnRefRawExpr *>(item.expr_);
        TableItem *ti = sel->get_table_item_by_id(col->get_table_id());
        if (NULL != ti && ti->is_values_table() && NULL != ti->values_table_def_
            && ObValuesTableDef::ACCESS_OBJ == ti->values_table_def_->access_type_) {
          const ObValuesTableDef *td = ti->values_table_def_;
          const int64_t col_idx = col->get_column_id() - OB_APP_MIN_COLUMN_ID;
          nullable = false;
          for (int64_t r = 0; r < td->row_cnt_; ++r) {
            const int64_t idx = r * td->column_cnt_ + col_idx;
            if (idx < td->access_objs_.count() && td->access_objs_.at(idx).is_null()) {
              nullable = true;
              break;
            }
          }
        }
      }
    }
    if (0 != i && OB_FAIL(sql.append(" union all "))) {
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(sql.append_fmt("select cast('%.*s' as char(256)) as column_name, "
                                      "cast('%s' as char(32)) as column_type, "
                                      "cast('%s' as char(8)) as `null`, "
                                      "cast(NULL as char(256)) as `key`, "
                                      "cast(NULL as char(256)) as `default`, "
                                      "cast(NULL as char(256)) as `extra`",
                                      col_name.length(), col_name.ptr(),
                                      type_name, nullable ? "YES" : "NO"))) {
    }
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("copy generated sql", "sql", sql.string());
    ParseResult parse_result;
    ObParser parser(*params.allocator_, params.session_info_->get_sql_mode());
    if (OB_FAIL(parser.parse(sql.string(), parse_result))) {
      LOG_WARN("failed to parse generated describe sql", K(ret), K(sql.string()));
    } else if (OB_ISNULL(parse_result.result_tree_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const ParseNode *stmt_node = parse_result.result_tree_;
      if (T_STMT_LIST == stmt_node->type_ && stmt_node->num_child_ > 0) {
        stmt_node = stmt_node->children_[0];
      }
      LOG_INFO("describe generated stmt node", "type", static_cast<int64_t>(stmt_node->type_));
      ObResolver res(params);
      if (OB_FAIL(res.resolve(ObResolver::IS_NOT_PREPARED_STMT, *stmt_node, stmt))) {
        LOG_WARN("failed to resolve generated describe sql", K(ret));
      }
    }
  }
  return ret;
}


static void describe_collect_opts(const ParseNode *node,
                                  common::ObIArray<common::ObString> &tokens)
{
  if (NULL == node) {
    return;
  }
  if (T_NAME_LIST == node->type_) {
    for (int64_t i = 0; i < node->num_child_; ++i) {
      describe_collect_opts(node->children_[i], tokens);
    }
  } else if (NULL != node->str_value_) {
    tokens.push_back(common::ObString(node->str_len_, node->str_value_));
  }
}

int ObResolver::resolve_copy_to(ObResolverParams &params,
                                const ParseNode &parse_tree,
                                ObStmt *&stmt)
{
  int ret = OB_SUCCESS;
  const ParseNode *sel_node = parse_tree.num_child_ > 0 ? parse_tree.children_[0] : NULL;
  const ParseNode *path_node = parse_tree.num_child_ > 1 ? parse_tree.children_[1] : NULL;
  const ParseNode *opt_node = parse_tree.num_child_ > 2 ? parse_tree.children_[2] : NULL;
  bool with_header = false;
  ObString null_str = ObString::make_string("NULL");
  ObString path;
  ObString inner_sql;
  ObStmt *child_stmt = NULL;
  ObSelectStmt *sel = NULL;
  ObSqlString sql;
  if (OB_ISNULL(sel_node) || OB_ISNULL(path_node) || NULL == path_node->str_value_) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    path.assign_ptr(path_node->str_value_, path_node->str_len_);
    ObSEArray<ObString, 8> tokens;
    describe_collect_opts(opt_node, tokens);
    {
      for (int64_t i = 0; OB_SUCC(ret) && i < tokens.count(); ++i) {
        LOG_INFO("copy opt token", "i", i, "tok", tokens.at(i));
        if (0 == tokens.at(i).case_compare("header")) {
          with_header = true;
        } else if (0 == tokens.at(i).case_compare("nullstr") && i + 1 < tokens.count()) {
          null_str = tokens.at(i + 1);
          ++i;
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    ObResolver inner(params);
    if (OB_FAIL(inner.resolve(ObResolver::IS_NOT_PREPARED_STMT, *sel_node, child_stmt))) {
    } else if (OB_ISNULL(child_stmt) || !child_stmt->is_select_stmt()) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      sel = static_cast<ObSelectStmt *>(child_stmt);
    }
  }
  // recover inner query text from the original sql
  if (OB_SUCC(ret)) {
    ObString orig = params.session_info_->get_current_query_string();
    if (orig.empty() && NULL != params.query_ctx_) {
      orig = params.query_ctx_->get_sql_stmt();
    }
    LOG_INFO("copy orig sql", "len", orig.length(), "text", orig);
    if (orig.empty()) {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "copy without original sql text");
    } else {
      int64_t begin = -1;
      for (int64_t i = 0; i < orig.length(); ++i) {
        if ('(' == orig.ptr()[i]) { begin = i + 1; break; }
      }
      int64_t depth = 0;
      int64_t end = -1;
      for (int64_t i = begin; begin > 0 && i < orig.length(); ++i) {
        const char c = orig.ptr()[i];
        if ('\'' == c || '"' == c) {
          const char q = c;
          ++i;
          while (i < orig.length() && !(q == orig.ptr()[i] && orig.ptr()[i - 1] != '\\')) { ++i; }
        } else if ('(' == c) {
          ++depth;
        } else if (')' == c) {
          if (0 == depth) { end = i; break; }
          --depth;
        }
      }
      if (begin <= 0 || end <= begin) {
        ret = OB_NOT_SUPPORTED;
      } else {
        inner_sql.assign_ptr(orig.ptr() + begin, static_cast<int32_t>(end - begin));
      }
    }
  }
  if (OB_SUCC(ret)) {
    if (OB_FAIL(sql.append("select * from ("))) {
    }
    if (OB_SUCC(ret) && with_header) {
      if (OB_FAIL(sql.append("select "))) {
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < sel->get_select_item_size(); ++i) {
        const SelectItem &item = sel->get_select_item(i);
        ObString col_name = item.alias_name_.empty() ? item.expr_name_ : item.alias_name_;
        if (OB_FAIL(sql.append_fmt("%s'%.*s'", 0 == i ? "" : ", ", col_name.length(), col_name.ptr()))) {
        }
      }
      if (OB_SUCC(ret) && OB_FAIL(sql.append(" union all "))) {
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(sql.append("select "))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < sel->get_select_item_size(); ++i) {
      const SelectItem &item = sel->get_select_item(i);
      ObString col_name = item.alias_name_.empty() ? item.expr_name_ : item.alias_name_;
      if (OB_FAIL(sql.append_fmt("%scoalesce(cast(`%.*s` as char),'%.*s')",
                                 0 == i ? "" : ", ",
                                 col_name.length(), col_name.ptr(),
                                 null_str.length(), null_str.ptr()))) {
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(sql.append_fmt(" from (%.*s) __copy_src) __copy_out "
                                               "into outfile '%.*s' "
                                               "fields terminated by ',' "
                                               "lines terminated by '\\n'",
                                               inner_sql.length(), inner_sql.ptr(),
                                               path.length(), path.ptr()))) {
    }
  }
  if (OB_SUCC(ret)) {
    ParseResult parse_result;
    ObParser parser(*params.allocator_, params.session_info_->get_sql_mode());
    if (OB_FAIL(parser.parse(sql.string(), parse_result))) {
      LOG_WARN("failed to parse generated copy sql", K(ret), K(sql.string()));
    } else if (OB_ISNULL(parse_result.result_tree_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const ParseNode *stmt_node = parse_result.result_tree_;
      if (T_STMT_LIST == stmt_node->type_ && stmt_node->num_child_ > 0) {
        stmt_node = stmt_node->children_[0];
      }
      ObResolver res(params);
      if (OB_FAIL(res.resolve(ObResolver::IS_NOT_PREPARED_STMT, *stmt_node, stmt))) {
        LOG_WARN("failed to resolve generated copy sql", K(ret), K(sql.string()));
      }
    }
  }
  return ret;
}

int ObResolver::resolve(IsPrepared if_prepared, const ParseNode &parse_tree, ObStmt *&stmt)
{
#define REGISTER_STMT_RESOLVER(name)                                   \
  do {                                                                 \
    ret = stmt_resolver_func<Ob##name##Resolver>(params_, *real_parse_tree, stmt); \
  } while (0)

#define REGISTER_SELECT_STMT_RESOLVER(name)                                   \
  do {                                                                        \
    ret = select_stmt_resolver_func<Ob##name##Resolver>(params_, parse_tree, stmt); \
  } while (0)
  ACTIVE_SESSION_FLAG_SETTER_GUARD(in_resolve);
  int ret = OB_SUCCESS;
  const ParseNode *real_parse_tree = NULL;
  int64_t questionmark_count = 0;
  UNUSED(if_prepared);
  if (OB_ISNULL(params_.allocator_)
      || OB_ISNULL(params_.schema_checker_)
      || OB_ISNULL(params_.session_info_)
      || OB_ISNULL(params_.expr_factory_)
      || OB_ISNULL(params_.query_ctx_)) {
    ret = OB_NOT_INIT;
  } else {
    real_parse_tree = &parse_tree;
  }

  if (OB_SUCC(ret)) {
    params_.resolver_scope_stmt_type_ = parse_tree.type_;
    params_.session_info_->set_stmt_type(
      ObResolverUtils::get_stmt_type_by_item_type(parse_tree.type_));
    ObExecContext *exec_ctx = params_.session_info_->get_cur_exec_ctx();
    if (exec_ctx != nullptr && exec_ctx->get_sql_ctx() != nullptr) {
      //save the current stmt type on my context,
      //some other modules may need to get the current SQL stmt type through its context
      ObSqlCtx *sql_ctx = exec_ctx->get_sql_ctx();
      sql_ctx->stmt_type_ = ObResolverUtils::get_stmt_type_by_item_type(parse_tree.type_);
    }

    switch (real_parse_tree->type_) {
      case T_CREATE_TABLE: {
        REGISTER_STMT_RESOLVER(CreateTable);
        break;
      }
      case T_ALTER_TABLE: {
        REGISTER_STMT_RESOLVER(AlterTable);
        break;
      }
      case T_CREATE_INDEX: {
        REGISTER_STMT_RESOLVER(CreateIndex);
        break;
      }
      case T_CREATE_VIEW: {
        REGISTER_STMT_RESOLVER(CreateView);
        break;
      }
      case T_ALTER_VIEW: {
#if 0
        ObAlterViewResolver stmt_resolver(params_);
        if (OB_FAIL(stmt_resolver.resolve(*node))) {
        }

        stmt = stmt_resolver.get_basic_stmt();
#endif
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("not support to alter view");
	      LOG_USER_ERROR(OB_NOT_SUPPORTED, "alter view");
        break;
      }
      case T_DROP_VIEW:
      case T_DROP_TABLE: {
        REGISTER_STMT_RESOLVER(DropTable);
        break;
      }
      case T_DROP_INDEX: {
        REGISTER_STMT_RESOLVER(DropIndex);
        break;
      }
      case T_CREATE_TABLE_LIKE: {
        REGISTER_STMT_RESOLVER(CreateTableLike);
        break;
      }
      case T_FORK_TABLE: {
        REGISTER_STMT_RESOLVER(ForkTable);
        break;
      }
      case T_FORK_DATABASE: {
        REGISTER_STMT_RESOLVER(ForkDatabase);
        break;
      }
      case T_DIFF_TABLE: {
        REGISTER_STMT_RESOLVER(DiffTable);
        break;
      }
      case T_MERGE_TABLE: {
        REGISTER_STMT_RESOLVER(MergeTable);
        break;
      }
      case T_SELECT: {
        REGISTER_SELECT_STMT_RESOLVER(Select);
        break;
      }
      case T_INSERT: {
        REGISTER_STMT_RESOLVER(Insert);
        break;
      }
      case T_UPDATE: {
        REGISTER_STMT_RESOLVER(Update);
        break;
      }
      case T_DELETE: {
        REGISTER_STMT_RESOLVER(Delete);
        break;
      }
      case T_BEGIN: {
        REGISTER_STMT_RESOLVER(StartTrans);
        break;
      }
      case T_ROLLBACK:
        //fall through
      case T_COMMIT: {
        REGISTER_STMT_RESOLVER(EndTrans);
        break;
      }
      case T_FREEZE: {
        REGISTER_STMT_RESOLVER(Freeze);
        break;
      }
      case T_FLUSH_CACHE: {
        REGISTER_STMT_RESOLVER(FlushCache);
        break;
      }
      case T_FLUSH_KVCACHE: {
        REGISTER_STMT_RESOLVER(FlushKVCache);
        break;
      }
      case T_FLUSH_ILOGCACHE: {
        REGISTER_STMT_RESOLVER(FlushIlogCache);
        break;
      }
      case T_FLUSH_DAG_WARNINGS: {
        REGISTER_STMT_RESOLVER(FlushDagWarnings);
        break;
      }
      case T_MERGE_CONTROL: {
        REGISTER_STMT_RESOLVER(AdminMerge);
        break;
      }
      case T_CANCEL_TASK: {
        REGISTER_STMT_RESOLVER(CancelTask);
        break;
      }
      case T_REFRESH_MEMORY_STAT: {
        REGISTER_STMT_RESOLVER(RefreshMemStat);
        break;
      }
      case T_REFRESH_IO_CALIBRATION: {
        REGISTER_STMT_RESOLVER(RefreshIOCalibration);
        break;
      }
      case T_SWITCHOVER_TO_STANDBY:
      case T_SWITCHOVER_TO_PRIMARY:
      case T_ACTIVATE_STANDBY: {
        REGISTER_STMT_RESOLVER(SwitchRole);
        break;
      }
      case T_ALTER_SYSTEM_SET_PARAMETER: {
        REGISTER_STMT_RESOLVER(SetConfig);
        break;
      }
      case T_ALTER_SYSTEM_SETTP: {
        REGISTER_STMT_RESOLVER(SetTP);
        break;
      }
      case T_CLEAR_MERGE_ERROR: {
        REGISTER_STMT_RESOLVER(ClearMergeError);
        break;
      }
      case T_CREATE_DATABASE: {
        REGISTER_STMT_RESOLVER(CreateDatabase);
        break;
      }
      case T_USE_DATABASE: {
        REGISTER_STMT_RESOLVER(UseDatabase);
        break;
      }
      case T_ALTER_DATABASE: {
        REGISTER_STMT_RESOLVER(AlterDatabase);
        break;
      }
      case T_DROP_DATABASE: {
        REGISTER_STMT_RESOLVER(DropDatabase);
        break;
      }
      case T_RENAME_TABLE: {
        REGISTER_STMT_RESOLVER(RenameTable);
        break;
      }
      case T_TRUNCATE_TABLE: {
        REGISTER_STMT_RESOLVER(TruncateTable);
        break;
      }
      case T_RECYCLEBIN_RESTORE_TABLE: {
        REGISTER_STMT_RESOLVER(RecyclebinRestoreTable);
        break;
      }
      case T_RECYCLEBIN_RESTORE_DATABASE: {
        REGISTER_STMT_RESOLVER(RecyclebinRestoreDatabase);
        break;
      }
      case T_PURGE_TABLE: {
        REGISTER_STMT_RESOLVER(PurgeTable);
        break;
      }
      case T_PURGE_INDEX: {
        REGISTER_STMT_RESOLVER(PurgeIndex);
        break;
      }
      case T_PURGE_DATABASE: {
        REGISTER_STMT_RESOLVER(PurgeDatabase);
        break;
      }
      case T_PURGE_RECYCLEBIN: {
        REGISTER_STMT_RESOLVER(PurgeRecycleBin);
        break;
      }
      case T_OPTIMIZE_TABLE: {
        REGISTER_STMT_RESOLVER(OptimizeTable);
        break;
      }
      case T_PREPARE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Prepare);
        }
        break;
      }
      case T_EXECUTE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Execute);
        }
        break;
      }
      case T_DEALLOCATE: {
        if (params_.is_prepare_protocol_) {
          ret = OB_ERR_PARSE_SQL;
        } else {
          REGISTER_STMT_RESOLVER(Deallocate);
        }
        break;
      }
      case T_VARIABLE_SET: {
        REGISTER_STMT_RESOLVER(VariableSet);
        break;
      }
      case T_TRANSACTION: {
        REGISTER_STMT_RESOLVER(SetTransaction);
        break;
      }
      case T_SHOW_TABLES:
      case T_SHOW_DATABASES:
      case T_SHOW_VARIABLES:
      case T_SHOW_COLUMNS:
      case T_SHOW_SCHEMA:
      case T_SHOW_CREATE_DATABASE:
      case T_SHOW_CREATE_TABLE:
      case T_SHOW_CREATE_VIEW:
      case T_SHOW_CREATE_PROCEDURE:
      case T_SHOW_CREATE_FUNCTION:
      case T_SHOW_TABLE_STATUS:
      case T_SHOW_PARAMETERS:
      case T_SHOW_INDEXES:
      case T_SHOW_PROCESSLIST:
      case T_SHOW_WARNINGS:
      case T_SHOW_ERRORS:
      case T_SHOW_TRACE:
      case T_SHOW_ENGINES:
      case T_SHOW_PROFILE:
      case T_SHOW_ENGINE:
      case T_SHOW_OPEN_TABLES:
      case T_SHOW_PRIVILEGES:
      case T_SHOW_CHARSET:
      case T_SHOW_COLLATION:
      case T_SHOW_GRANTS:
      case T_SHOW_RECYCLEBIN:
      case T_SHOW_PROCEDURE_STATUS:
      case T_SHOW_FUNCTION_STATUS:
      case T_SHOW_TRIGGERS:
      case T_SHOW_STATUS:
      case T_SHOW_CREATE_TRIGGER:
      case T_SHOW_CHECK_TABLE:
      case T_SHOW_CREATE_USER: {
        REGISTER_STMT_RESOLVER(Show);
        break;
      }
      case T_CREATE_USER: {
        REGISTER_STMT_RESOLVER(CreateUser);
        break;
      }
      case T_DROP_USER: {
        REGISTER_STMT_RESOLVER(DropUser);
        break;
      }
      case T_RENAME_USER: {
        REGISTER_STMT_RESOLVER(RenameUser);
        break;
      }
      case T_SET_PASSWORD: {
        REGISTER_STMT_RESOLVER(SetPassword);
        break;
      }
      case T_LOCK_USER: {
        REGISTER_STMT_RESOLVER(LockUser);
        break;
      }
      case T_ALTER_USER_DEFAULT_ROLE:
      case T_SET_ROLE: {
        REGISTER_STMT_RESOLVER(AlterUserRole);
        break;
      }

      case T_GRANT_ROLE:
      case T_SYSTEM_GRANT:
      case T_GRANT: {
        REGISTER_STMT_RESOLVER(Grant);
        break;
      }
      case T_SYSTEM_REVOKE:
      case T_REVOKE:
      case T_REVOKE_ALL: {
        REGISTER_STMT_RESOLVER(Revoke);
        break;
      }
      case T_EXPLAIN: {
        REGISTER_STMT_RESOLVER(Explain);
        break;
      }
      case T_DESCRIBE_SELECT: {
        ret = ObResolver::resolve_describe_select(params_, *real_parse_tree, stmt);
        break;
      }
      case T_COPY_TO: {
        ret = ObResolver::resolve_copy_to(params_, *real_parse_tree, stmt);
        break;
      }
      case T_ALTER_SYSTEM_SET: {
        REGISTER_STMT_RESOLVER(AlterSystemSet);
        break;
      }
      case T_ALTER_SYSTEM_KILL: {
        REGISTER_STMT_RESOLVER(Kill);
        break;
      }
      case T_ALTER_SESSION_SET: {
        REGISTER_STMT_RESOLVER(AlterSessionSet);
        break;
      }
      case T_SET_NAMES:
        // fall through
      case T_SET_CHARSET: {
        REGISTER_STMT_RESOLVER(SetNames);
        break;
      }
      case T_KILL: {
        REGISTER_STMT_RESOLVER(Kill);
        break;
      }
      case T_EMPTY_QUERY: {
        REGISTER_STMT_RESOLVER(EmptyQuery);
        break;
      }
      case T_FLUSH_PRIVILEGES: {
        REGISTER_STMT_RESOLVER(EmptyQuery);
        break;
      }
      case T_LOCK_TABLE: {
        REGISTER_STMT_RESOLVER(LockTable);
        break;
      }
      case T_CREATE_OUTLINE: {
        REGISTER_STMT_RESOLVER(CreateOutline);
        break;
      }
      case T_ALTER_OUTLINE: {
        REGISTER_STMT_RESOLVER(AlterOutline);
        break;
      }
      case T_DROP_OUTLINE: {
        REGISTER_STMT_RESOLVER(DropOutline);
        break;
      }
      case T_SP_CREATE: {
        REGISTER_STMT_RESOLVER(CreateProcedure);
        break;
      }
      case T_SP_ALTER: {
        REGISTER_STMT_RESOLVER(AlterProcedure);
        break;
      }
      case T_SP_CALL_STMT: {
        REGISTER_STMT_RESOLVER(CallProcedure);
        break;
      }
      case T_SF_CREATE: {
        REGISTER_STMT_RESOLVER(CreateFunction);
        break;
      }
      case T_SF_ALTER: {
        REGISTER_STMT_RESOLVER(AlterFunction);
        break;
      }
      case T_SP_DROP: {
        REGISTER_STMT_RESOLVER(DropProcedure);
        break;
      }
      case T_SF_DROP: {
        REGISTER_STMT_RESOLVER(DropFunction);
        break;
      }
      case T_SP_ANONYMOUS_BLOCK: {
        REGISTER_STMT_RESOLVER(AnonymousBlock);
        break;
      }
      case T_PACKAGE_CREATE: {
        REGISTER_STMT_RESOLVER(CreatePackage);
        break;
      }
      case T_PACKAGE_CREATE_BODY: {
        REGISTER_STMT_RESOLVER(CreatePackageBody);
        break;
      }
      case T_PACKAGE_DROP: {
        REGISTER_STMT_RESOLVER(DropPackage);
        break;
      }
      case T_MYSQL_UPDATE_HISTOGRAM:
      case T_MYSQL_DROP_HISTOGRAM:
      case T_MYSQL_ANALYZE: {
        REGISTER_STMT_RESOLVER(AnalyzeStmt);
        break;
      }
      case T_LOAD_DATA_URL:
      case T_LOAD_DATA: {
        REGISTER_STMT_RESOLVER(LoadData);
        break;
      }
      case T_SET_TABLE_COMMENT:
      case T_SET_COLUMN_COMMENT: {
        REGISTER_STMT_RESOLVER(SetComment);
        break;
      }
      case T_CREATE_ROLE: {
        REGISTER_STMT_RESOLVER(CreateRole);
        break;
      }
      case T_DROP_ROLE: {
        REGISTER_STMT_RESOLVER(DropRole);
        break;
      }
      case T_ALTER_ROLE: {
        REGISTER_STMT_RESOLVER(AlterRole);
        break;
      }
      case T_CREATE_SAVEPOINT:
      case T_ROLLBACK_SAVEPOINT:
      case T_RELEASE_SAVEPOINT: {
        REGISTER_STMT_RESOLVER(SavePoint);
        break;
      }
      case T_TG_CREATE: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_TG_DROP: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_TG_ALTER: {
        REGISTER_STMT_RESOLVER(Trigger);
        break;
      }
      case T_DIAGNOSTICS: {
        REGISTER_STMT_RESOLVER(GetDiagnostics);
        break;
      }
      case T_ALTER_SYSTEM_RESET_PARAMETER: {
        REGISTER_STMT_RESOLVER(ResetConfig);
        break;
      }
      case T_ALTER_SYSTEM_RESET: {
        REGISTER_STMT_RESOLVER(AlterSystemReset);
        break;
      }
      default: {
        ret = OB_NOT_SUPPORTED;
        LOG_INFO("unsupported resolve stmt type", "item_type", static_cast<int64_t>(parse_tree.type_));
        const char *type_name = get_type_name(parse_tree.type_);
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "statement type");
        break;
      }
    }  // end switch
    if (OB_SUCC(ret) && stmt->is_dml_write_stmt()) {
      // todo yanli:check leader-follower database
    }
    
    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !params_.session_info_->is_varparams_sql_prepare()) {
      ObDMLStmt *dml_stmt = static_cast<ObDMLStmt*>(stmt);
      ObRawExprWrapEnumSet enum_set_wrapper(*params_.expr_factory_, params_.session_info_);
      if (OB_FAIL(enum_set_wrapper.wrap_enum_set(*dml_stmt))) {
      }
    }

    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !stmt->is_explain_stmt()) {
      if (OB_FAIL(params_.query_ctx_->query_hint_.init_query_hint(params_.allocator_,
                                                                  params_.session_info_,
                                                                  static_cast<ObDMLStmt*>(stmt)))) {
      } else if (OB_FAIL(params_.query_ctx_->query_hint_.check_and_set_params_from_hint(params_,
                                                         *static_cast<ObDMLStmt*>(stmt)))) {
      }
    }

    if (OB_SUCC(ret) && stmt->is_dml_stmt() && !stmt->is_explain_stmt()) {
      bool is_contain_inner_table = false;
      bool is_contain_select_for_update = false;
      ObDMLStmt *dml_stmt = static_cast<ObDMLStmt*>(stmt);
      if (OB_FAIL(dml_stmt->check_if_contain_inner_table(is_contain_inner_table))) {
      } else if (OB_FAIL(dml_stmt->check_if_contain_select_for_update(
                  is_contain_select_for_update))) {
      } else {
        params_.query_ctx_->is_contain_inner_table_ = is_contain_inner_table;
        params_.query_ctx_->is_contain_select_for_update_ = is_contain_select_for_update;
        params_.query_ctx_->has_dml_write_stmt_ = dml_stmt->is_dml_write_stmt();
      }

    }
    if (OB_SUCC(ret)) {
      stmt::StmtType stmt_type = stmt->get_stmt_type();
      if (ObStmt::is_ddl_stmt(stmt_type, stmt->has_global_variable()) || ObStmt::is_dcl_stmt(stmt_type)) {
        ObDDLStmt *ddl_stmt = static_cast<ObDDLStmt*>(stmt);
        obcall::ObDDLArg &ddl_arg = ddl_stmt->get_ddl_arg();

        if (OB_ISNULL(params_.query_ctx_)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          ddl_arg.ddl_stmt_str_ = params_.query_ctx_->get_sql_stmt();
        }
      }
    }
  }  // end if
  return ret;
}

}  // namespace sql
}  // namespace oceanbase

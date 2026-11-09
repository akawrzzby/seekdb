#define USING_LOG_PREFIX SERVER
#include "observer/virtual_table/ob_all_ai_column_task_virtual_table.h"
using namespace oceanbase::common;
namespace oceanbase
{
namespace observer
{
int ObAllAiColumnTaskVirtualTable::inner_get_next_row(common::ObNewRow *&row)
{
  UNUSED(row);
  return OB_ITER_END;
}
}  // namespace observer
}  // namespace oceanbase

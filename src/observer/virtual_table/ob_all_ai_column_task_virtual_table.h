#ifndef OCEANBASE_OBSERVER_VIRTUAL_TABLE_OB_ALL_AI_COLUMN_TASK_VIRTUAL_TABLE_
#define OCEANBASE_OBSERVER_VIRTUAL_TABLE_OB_ALL_AI_COLUMN_TASK_VIRTUAL_TABLE_
#include "observer/virtual_table/ob_virtual_table_scanner_iterator.h"
namespace oceanbase
{
namespace observer
{
// Pending AI column tasks.  The task rows are produced by the AI column
// scheduler; until a task exists the table simply yields no row.
class ObAllAiColumnTaskVirtualTable : public common::ObVirtualTableScannerIterator
{
public:
  ObAllAiColumnTaskVirtualTable() {}
  virtual ~ObAllAiColumnTaskVirtualTable() { reset(); }
  virtual void reset() override { ObVirtualTableScannerIterator::reset(); }
  virtual int inner_get_next_row(common::ObNewRow *&row) override;
};
}  // namespace observer
}  // namespace oceanbase
#endif

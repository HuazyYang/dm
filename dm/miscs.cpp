#include "miscs.h"
#include "dm_miscs_double_ispc.h"
#include "dm_miscs_float_ispc.h"

namespace dm {

void print_double_isa() {
  ispc::print_double_isa();
}

void print_float_isa() {
  ispc::print_float_isa();
}

}
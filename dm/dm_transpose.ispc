#include "dm_simd_config.isph"

static
void _extended_gcd(uniform int a, uniform int b, uniform int &gcd, uniform int &mmi) {

  uniform int<2> xy = { 0, 1 };
  uniform int<2> last_xy = { 1, 0 };
  uniform int origb = b;

  while(b != 0) {
    uniform int quotient = a / b;
    uniform int newb = a % b;
    a = b;
    b = newb;

    uniform int<2> new_xy = last_xy - quotient * xy;
    last_xy = xy;
    xy = new_xy;
  }

  gcd = a;
  mmi = 0;
  if(gcd == 1) {
    if(last_xy.x < 0)
      mmi = last_xy.x + origb;
    else
      mmi = last_xy.x;
  }
}

static
uniform int _find_log_2(uniform int x, uniform bool round_up) {
  uniform int a = 31 - count_leading_zeros(x);
  if(round_up) a += !((x & (x - 1)) == 0);
  return a;
}

static
uniform int _find_log_2(uniform int64 x, uniform bool round_up) {
  uniform int a = 63 - count_leading_zeros(x);
  if(round_up) a += !((x & (x-1)) == 0);
  return a;
}

static
void _find_devisor(
  uniform uint32 denom,
  uniform uint32 &mul_coeff,
  uniform uint32 &shift_coeff
) {
  if(denom == 1) {
    mul_coeff = 0;
    shift_coeff = 0;
    return;
  }

  uniform int32 p = 31 + _find_log_2((uniform int32)denom, true);
  uniform int32 m = ((1ull << p) + denom - 1) / denom;
  mul_coeff = m;
  shift_coeff = p - 32;
}

/*
// TODO: _find_divisor support 64bit(large) integer
void _find_divisor(
  uniform uint64 denom,
  uniform uint64 &mul_coeff,
  uniform uint64 &shift_coef
) {

}
*/

static
uint32 _umulhi(uint32 x, uint32 y) {
  uint64 z = (uint64)x * (uint64)y;
  return (uint32)(z >> 32);
}

/*
// TODO: _umulhi support 64bit(large) integer
uint64 _umulhi(uint64 x, uint64 y) {
  __uint128_t z = (__uint128_t)x * (__uint128_t)y;
  return (uint64)(z >> 64);
}
*/

static
struct _reduced_divisor_32 {
  uint32 mul_coeff;
  uint32 shift_coeff;
  uint32 y;
};

static
void _reduced_divisor_32_init(uniform _reduced_divisor_32 &self, uniform uint32 y) {
  self.y = y;
  _find_devisor(self.y, self.mul_coeff, self.shift_coeff);
}

static
uint32 _reduced_divisor_32_div(uniform const _reduced_divisor_32 &self, uint32 x) {
  return self.mul_coeff ? _umulhi(x, self.mul_coeff) >> self.shift_coeff : x; 
}

static
uint32 _reduced_divisor_32_mod(uniform const _reduced_divisor_32 &self, uint32 x) {
  return self.mul_coeff ? x - _reduced_divisor_32_div(self, x) * self.y : 0;
}

static
void _reduced_divisor_32_divmod(uniform const _reduced_divisor_32 &self, uint32 x, uint32 &q, uint32 &mod) {
  if(self.y == 1) {
    q = x; mod = 0;
  } else {
    q = _reduced_divisor_32_div(self, x);
    mod = x - (q * self.y);
  }
}

static
uint32 _reduced_divisor_32_get(uniform const _reduced_divisor_32 &self) {
  return self.y;
}

struct _prerotator {
  _reduced_divisor_32 m, b;
  int x[programCount];
};

static void _prerotator_init(uniform _prerotator *uniform self, uniform int m, uniform int b) {
  _reduced_divisor_32_init(self->m, m);
  _reduced_divisor_32_init(self->b, b);
}
static uniform int _prerotator_set_j(uniform _prerotator* uniform self, int j) {
  int x = _reduced_divisor_32_div(self->b, j);
  return packed_store_active(self->x, x);
}
static int _prerotator_advance(uniform _prerotator* uniform self, uniform int idx, int i) {
  return _reduced_divisor_32_mod(self->m, self->x[idx] + i);
}

struct _postpermuter {
  _reduced_divisor_32 m;
  int n;
  _reduced_divisor_32 a;
  int j[programCount];
};
static void _postpermuter_init(uniform _postpermuter* uniform self, uniform int m, uniform int n, uniform int a) {
  _reduced_divisor_32_init(self->m, m);
  self->n = n;
  _reduced_divisor_32_init(self->a, a);
}
static uniform int _postpermuter_set_j(uniform _postpermuter* uniform self, int j) {
  return packed_store_active(self->j, j);
}
static int _postpermuter_advance(uniform _postpermuter* uniform self, uniform int idx, int i) {
  return _reduced_divisor_32_mod(
    self->m,
    (i * self->n + self->j[idx] - _reduced_divisor_32_div(self->a, i))
  );
}

struct _shuffle {
  int m, n, k;
  _reduced_divisor_32 b;
  _reduced_divisor_32 c;
  int i[programCount];
};
static void _shuffle_init(uniform _shuffle* uniform self, uniform int m, uniform int n, uniform int c, uniform int k) {
  self->m = m;
  self->n = n;
  self->k = k;
  _reduced_divisor_32_init(self->b, n / c);
  _reduced_divisor_32_init(self->c, c);
}
static uniform int _shuffle_set_i(uniform _shuffle* uniform self, int i) {
  return packed_store_active(self->i, i);
}
static int _shuffle_f(uniform _shuffle* uniform self, uniform int idx, int j) {
  int r = j + (self->i[idx] * (self->n - 1));
  if(self->i[idx] - (int)_reduced_divisor_32_mod(self->c, j)
    <= self->m  - (int)_reduced_divisor_32_get(self->c))
    return r;
  else
    return r + self->m;
}
static int _shuffle_advance(uniform _shuffle* uniform self, uniform int idx, int j) {
  int fij = _shuffle_f(self, idx, j);
  //The extra mod in here prevents overflowing 32-bit int
  uint fijdivc, fijmodc;
  _reduced_divisor_32_divmod(self->c, fij, fijdivc, fijmodc);
  int term_1 = _reduced_divisor_32_mod(self->b, 
    self->k * _reduced_divisor_32_mod(self->b, fijdivc));
  int term_2 = ((int)fijmodc) * (int)_reduced_divisor_32_get(self->b);
  return term_1 + term_2;
}

typedef uniform int (*_shuffle_callback_set_ij_t)(void * uniform ctx, int ij);
typedef int(*_shuffle_callback_advance_t)(void *uniform ctx, uniform int idx, int ji);

struct _shuffle_callback {
  void *ctx;
  _shuffle_callback_set_ij_t set_ij;
  _shuffle_callback_advance_t advance;
};

#if TARGET_WIDTH == 8
void _transpose_block(
  real block[]
) {
  uniform int P[3][8] = {
    { 0, 8, 2, 10, 4, 12, 6, 14 },
    { 0, 1, 8, 9, 4, 5, 12, 13 },
    { 0, 1, 2, 3, 8, 9, 10, 11 }, };

  real tmp;
  int p;

  p = P[0][programIndex];
  for(uniform int i = 0; i < 8; i += 2) {
    tmp = shuffle(block[i], block[i+1], p);
    block[i+1] = shuffle(block[i], block[i+1], p+1);
    block[i] = tmp;
  }

  p = P[1][programIndex];
  for(uniform int i = 0; i < 8; i += 3) {
    tmp = shuffle(block[i], block[i+2], p);
    block[i+2] = shuffle(block[i], block[i+2], p+2);
    block[i] = tmp;

    i += 1;
    tmp = shuffle(block[i], block[i+2], p);
    block[i+2] = shuffle(block[i], block[i+2], p+2);
    block[i] = tmp;
  }

  p = P[2][programIndex];
  for(uniform int i = 0; i < 4; ++i) {
    tmp = shuffle(block[i], block[i+4], p);
    block[i+4] = shuffle(block[i], block[i+4], p+4);
    block[i] = tmp;
  }
}
#elif TARGET_WIDTH == 4
void _transpose_block(
  real block[]
) {
  uniform int P[2][4] = {
    { 0, 4, 2, 6 },
    { 0, 1, 4, 5 }
  };
  real tmp;
  int p;

  p = P[0][programIndex];
  tmp = shuffle(block[0], block[1], p);
  block[1] = shuffle(block[0], block[1], p+1);
  block[0] = tmp;

  tmp = shuffle(block[2], block[3], p);
  block[3] = shuffle(block[2], block[3], p+1);
  block[2] = tmp;

  p = P[1][programIndex];
  tmp = shuffle(block[0], block[2], p);
  block[2] = shuffle(block[0], block[2], p+2);
  block[0] = tmp;

  tmp = shuffle(block[1], block[3], p);
  block[3] = shuffle(block[1], block[3], p+2);
  block[1] = tmp;
}
#else
  #error Unsupported lane width
#endif

void _col_shuffle(
  uniform int m,
  uniform int n,
  uniform real data[],
  varying real tmp[],
  uniform _shuffle_callback * uniform fn
) {
  uniform int jbases[programCount];
  uniform int count;
  uniform int ind;
  real block[programCount];

  for(int j = programIndex; j < n; j += programCount) {
    jbases[programIndex] = j;
    count = fn->set_ij(fn->ctx, j);
    ind = 0;
    foreach(i = 0 ... m) {
      for(uniform int k = 0; k < count; ++k) {
        int i2 = fn->advance(fn->ctx, k, i);
        block[k] = data[i2 * n + jbases[k]];
      }
      // inverse block
      unmasked {
        _transpose_block(block);
        memcpy(tmp + ind, block, min(programCount, m - ind) * sizeof(block[0]));
        ind += programCount;
      }
    }
    for(uniform int i = 0; i < m; ++i) {
      real val = tmp[i];
      uniform int ibase = i*n;
      data[ibase+j] = val;
    }
  }
}

void _row_shuffle(
  uniform int m,
  uniform int n,
  uniform real data[],
  varying real tmp[],
  uniform _shuffle_callback *uniform fn
) {
  uniform int ibases[programCount];
  uniform int count;
  uniform int ind;
  varying real block[programCount];

  for(int i = programIndex; i < m; i += programCount) {
    int ibase = i*n;
    ibases[programIndex] = ibase;
    count = fn->set_ij(fn->ctx, i);
    ind = 0;
    foreach(j = 0 ... n) {
      for(uniform int k = 0; k < count; ++k) {
        int j2 = fn->advance(fn->ctx, k, j);
        block[k] = data[ibases[k] + j2];
      }
      // inverse block
      unmasked {
        _transpose_block(block);
        memcpy(tmp + ind, block, min(programCount, n - ind) * sizeof(block[0]));
        ind += programCount;
      }
    }
    for(uniform int j = 0; j < n; ++j) {
      real val = tmp[j];
      data[ibase+j] = val;
    }
  }
}

inline
const void *uniform fnptr_cast_pvoid(const void *uniform p) {
  return p;
}

void _mat_transpose_inplace(
  uniform int m,
  uniform int n,
  uniform real data[]
) {
  uniform int c, t, k;
  _extended_gcd(m, n, c, t);
  if(c > 1)
    _extended_gcd(m/c, n/c, t, k);
  else
    k = t;

  uniform bool tmp_stack_alloc;
  uniform int tmp_count = max(m, n);
  varying real* uniform tmp;
  if(tmp_count * sizeof(real) < DM_SIMD_ALLOCA_THRESHOLD) {
    tmp_stack_alloc = true;
    tmp = (varying real *uniform)alloca(tmp_count * sizeof(real));
  } else {
    tmp_stack_alloc = false;
    tmp = uniform new varying real[tmp_count];
  }

  if(c > 1) {
    uniform _prerotator rotator;
    _prerotator_init(&rotator, m, n/c);
    uniform _shuffle_callback fn = {
      &rotator,
      (_shuffle_callback_set_ij_t)fnptr_cast_pvoid(_prerotator_set_j),
      (_shuffle_callback_advance_t)fnptr_cast_pvoid(_prerotator_advance)
    };
    _col_shuffle(m, n, data, tmp, &fn);
  }

  {
    uniform _shuffle permuter;
    _shuffle_init(&permuter, m, n, c, k);
    uniform _shuffle_callback fn = {
      &permuter,
      (_shuffle_callback_set_ij_t)fnptr_cast_pvoid(_shuffle_set_i),
      (_shuffle_callback_advance_t)fnptr_cast_pvoid(_shuffle_advance)
    };
    _row_shuffle(m, n, data, tmp, &fn);
  }

  {
    uniform _postpermuter permuter;
    _postpermuter_init(&permuter, m, n, m/c);
    uniform _shuffle_callback fn = {
      &permuter,
      (_shuffle_callback_set_ij_t)fnptr_cast_pvoid(_postpermuter_set_j),
      (_shuffle_callback_advance_t)fnptr_cast_pvoid(_postpermuter_advance)
    };

    _col_shuffle(m, n, data, tmp, &fn);
  }

  if(!tmp_stack_alloc)
    delete[] tmp;
  return;
}

export
void REAL_FN(mat_transpose)(
  uniform real data[],
  uniform int m,
  uniform int n
) {
  _mat_transpose_inplace(m, n, data);
}


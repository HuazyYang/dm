#ifndef PDF_H
#define PDF_H

#include <random>

extern
unsigned long long get_random_seed();

template <typename T> class uniform_real_pdf {
public:
  uniform_real_pdf(T xmin, T xmax) : dre(get_random_seed()), pdf{xmin, xmax} {
  }

  T operator()() {
    return pdf(dre);
  }

  void set_range(T xmin, T xmax) {
    pdf.param(std::uniform_real_distribution<T>::param_type{xmin, xmax});
  }

private:
  std::default_random_engine dre;
  std::uniform_real_distribution<T> pdf;
};

template <typename T> class uniform_int_pdf {
public:
  uniform_int_pdf(T xmin, T xmax) : dre(get_random_seed()), pdf{xmin, xmax} {
    std::random_device rdev;
    dre.seed(rdev());
  }

  T operator()() {
    return pdf(dre);
  }

  void set_range(T xmin, T xmax) {
    pdf.param(std::uniform_real_distribution<T>::param_type{xmin, xmax});
  }

private:
  std::default_random_engine dre;
  std::uniform_int_distribution<T> pdf;
};

#endif /* PDF_H */

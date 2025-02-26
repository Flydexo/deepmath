#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct
{
    double re;
    double i;
} complex;

typedef struct
{
    int len;
    complex *coeffs;
} Poly;

typedef struct
{
    int len;
    complex *values;
} Evals;

complex c_add(complex a, complex b)
{
    complex c = {a.re + b.re, a.i + b.i};
    return c;
}

complex c_times(complex a, complex b)
{
    complex c = {a.re * b.re - a.i * b.i, a.i * b.re + a.re * b.i};
    return c;
}

complex c_pow(complex a, int n)
{
    if (n == 0)
    {
        complex c = {1, 0};
        return c;
    }
    return c_times(a, c_pow(a, n - 1));
}

// for even b = 0,
// for odd b = 1
Poly extract_coeffs(Poly p, int b)
{
    complex *p_coeffs = malloc(sizeof(complex) * (p.len / 2));
    for (int i = 0; i < p.len / 2; i++)
    {
        p_coeffs[i] = p.coeffs[2 * i + b];
    }
    Poly p_e = {p.len / 2, p_coeffs};
    return p_e;
}

Evals FFT(Poly p)
{
    assert(fmod(log2(p.len), 1) == 0);
    if (p.len == 1)
    {
        Evals y = {p.len, p.coeffs};
        return y;
    }
    Evals pe = FFT(extract_coeffs(p, 0));
    Evals po = FFT(extract_coeffs(p, 1));
    Evals y = {p.len, malloc(sizeof(complex) * p.len)};
    complex n_root = {cos(2 * M_PI / p.len), sin(2 * M_PI / p.len)};
    for (int i = 0; i < p.len / 2; i++)
    {
        complex omega_i = c_pow(n_root, i);
        y.values[i] = c_add(pe.values[i], c_times(omega_i, po.values[i]));
        complex minus_one = {-1, 0};
        y.values[i + p.len / 2] = c_add(pe.values[i], c_times(minus_one, c_times(omega_i, po.values[i])));
    }
    free(pe.values);
    free(po.values);
    return y;
}

int main()
{
    Poly p = {4, malloc(sizeof(complex) * 4)};

    p.coeffs[0].re = 1;
    p.coeffs[0].i = 0;
    p.coeffs[1].re = 2;
    p.coeffs[1].i = 0;
    p.coeffs[2].re = 3;
    p.coeffs[2].i = 0;
    p.coeffs[3].re = 4;
    p.coeffs[3].i = 0;

    Evals y = FFT(p);

    for (int i = 0; i < y.len; i++)
    {
        printf("%f + i%f\n", y.values[i].re, y.values[i].i);
    }

    free(y.values);
    free(p.coeffs);

    return 0;
}
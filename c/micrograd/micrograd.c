#include <stdlib.h>
#include <math.h>
#include "micrograd.h"

const Value ZERO = {0, 0, false, NULL, 0, NULL};

Value *v_init(double data)
{
    Value v = {data, 0, false, NULL, 0, NULL};
    Value *p = malloc(sizeof(Value));
    *p = v;
    return p;
}

Value *v_alloc(Value v)
{
    Value *p = malloc(sizeof(Value));
    *p = v;
    return p;
}

void add_backward(Value *self, Value *other, Value *out)
{
    self->grad += out->grad;
    other->grad += out->grad;
}

Value *v_add(Value *self, Value *other)
{
    Value out = {self->data + other->data, 0, false, malloc(sizeof(Value *) * 2), 2, add_backward};
    out.prev[0] = self;
    out.prev[1] = other;
    return v_alloc(out);
}

void mult_backward(Value *self, Value *other, Value *out)
{
    self->grad += other->data * out->grad;
    other->grad += self->data * out->grad;
}

Value *v_mult(Value *self, Value *other)
{
    Value out = {self->data * other->data, 0, false, malloc(sizeof(Value *) * 2), 2, mult_backward};
    out.prev[0] = self;
    out.prev[1] = other;
    return v_alloc(out);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void inv_backward(Value *self, Value *_, Value *out)
{
    self->grad += -1 / (self->data * self->data) * out->grad;
}

Value *v_inv(Value *self)
{
    Value out = {1 / self->data, 0, false, malloc(sizeof(Value *)), 1, inv_backward};
    out.prev[0] = self;
    return v_alloc(out);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void exp_backward(Value *self, Value *_, Value *out)
{
    self->grad += out->data * out->grad;
}

Value *v_exp(Value *self)
{
    Value out = {exp(self->data), 0, false, malloc(sizeof(Value *)), 1, exp_backward};
    out.prev[0] = self;
    return v_alloc(out);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void log_backward(Value *self, Value *_, Value *out)
{
    self->grad += (1 / self->data) * out->grad;
}

Value *v_log(Value *self)
{
    Value out = {log(self->data), 0, false, malloc(sizeof(Value *)), 1, log_backward};
    out.prev[0] = self;
    return v_alloc(out);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void sigmoid_backward(Value *self, Value *_, Value *out)
{
    self->grad += out->data * (1 - out->data) * out->grad;
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
Value *v_sigmoid(int _, Value **self, int __)
{
    return v_inv(v_add(v_init(1), v_exp(v_mult(v_init(-1), self[0]))));
}

Value *v_sum(Value **x, int len)
{
    if (len == 0)
    {
        return v_alloc(ZERO);
    }
    return v_add(x[0], v_sum(&x[1], len - 1));
}

Value **v_map(Value **x, int len, Value *(*f)(Value *))
{
    Value **y = malloc(sizeof(sizeof(Value)) * len);
    for (int i = 0; i < len; i++)
    {
        y[i] = f(x[i]);
    }
    return y;
}

Value *v_softmax(int i, Value **others, int len)
{
    Value **e = v_map(others, len, v_exp);
    Value *denominator = v_sum(e, len);
    Value **y = malloc(sizeof(Value) * len);
    for (int i = 0; i < len; i++)
    {
        Value *inv = v_inv(denominator);
        y[i] = v_mult(e[i], inv);
    }
    return y[i];
}

void v_backward(Value *self)
{
    self->grad = 1;
    if (self->prev_len != 0 || !self->visited)
    {
        for (int i = 0; i < self->prev_len; i++)
        {
            v_backward(self->prev[i]);
        }
    }
}

void v_reset_grad(Value *self)
{
    self->visited = false;
    self->grad = 0;
    if (self->prev_len != 0)
    {
        for (int i = 0; i < self->prev_len; i++)
        {
            v_reset_grad(self->prev[i]);
        }
    }
}
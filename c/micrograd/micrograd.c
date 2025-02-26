#include <stdlib.h>
#include <math.h>
#include "../ia-math/ia-math.h"
#include "micrograd.h"

const Value ZERO = {0, 0, NULL, NULL};

Value v_init(double data)
{
    Value v = {data, 0, NULL, NULL};
    return v;
}

void add_backward(Value *self, Value *other, Value *out)
{
    self->grad += out->grad;
    other->grad += out->grad;
}

Value v_add(Value *self, Value *other)
{
    Value out = {self->data + other->data, 0, malloc(sizeof(Value *) * 2), add_backward};
    return out;
}

void mult_backward(Value *self, Value *other, Value *out)
{
    self->grad += other->data * out->grad;
    other->grad += self->data * out->grad;
}

Value v_mult(Value *self, Value *other)
{
    Value out = {self->data * other->data, 0, malloc(sizeof(Value *) * 2), mult_backward};
    return out;
}

void inv_backward(Value *self, void *, Value *out)
{
    self->grad = -1 / (self->data * self->data) * out->grad;
}

Value v_inv(Value *self)
{
    Value out = {1 / self->data, 0, malloc(sizeof(Value *)), inv_backward};
    return out;
}

void exp_backward(Value *self, void *, Value *out)
{
    self->grad = out->data * out->grad;
}

Value v_exp(Value *self)
{
    Value out = {exp(self->data), 0, malloc(sizeof(Value *)), exp_backward};
    return out;
}

void sigmoid_backward(Value *self, void *, Value *out)
{
    self->grad = out->data * (1 - out->data) * out->grad;
}

Value v_sigmoid(Value *self)
{
    Value out = {sigmoid(self->data), 0, malloc(sizeof(Value *)), sigmoid_backward};
    return out;
}

Value v_sum(Value **x, int len)
{
    if (len == 0)
    {
        return ZERO;
    }
    return v_add(x[0], v_vum(&x[1], len - 1));
}

Value *v_map(Value **x, int len, Value (*f)(Value *))
{
    Value *y = malloc(sizeof(sizeof(Value)) * len);
    for (int i = 0; i < len; i++)
    {
        y[i] = f(x[i]);
    }
    return y;
}

Value *v_softmax(Value **others, int len)
{
    Value *e = v_map(others, len, v_exp);
    Value denominator = v_sum(e, len);
    Value *y = malloc(sizeof(Value) * len);
    for (int i = 0; i < len; i++)
    {
        Value inv = v_inv(&denominator);
        y[i] = v_mult(&e[i], &inv);
    }
    return y;
}
#include <stdbool.h>

typedef struct Value
{
    double data;
    double grad;
    bool visited;
    struct Value **prev;
    int prev_len;
    void (*backward)(struct Value *, struct Value *, struct Value *);
} Value;

const Value ZERO;

Value *v_init(double data);

Value *v_add(Value *self, Value *other);

Value *v_mult(Value *self, Value *other);

Value *v_exp(Value *self);

Value *v_log(Value *self);

Value *v_sigmoid(int i, Value **self, int len);
Value *v_softmax(int i, Value **self, int len);

void v_backward(Value *self);

void v_reset_grad(Value *self);
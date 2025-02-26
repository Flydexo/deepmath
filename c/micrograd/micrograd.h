typedef struct
{
    double data;
    double grad;
    Value **prev;
    void (*backward)(Value *, Value *, Value *);
} Value;

const Value ZERO;

Value v_init(double data);

Value v_add(Value *self, Value *other);

Value v_mult(Value *self, Value *other);

Value v_exp(Value *self);

Value v_sigmoid(Value *self);
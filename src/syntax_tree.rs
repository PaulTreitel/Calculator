use std::fmt::Display;

use super::tokenizer;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Parens(Option<Box<Expr>>),
    Number(Value),
    Add(Option<Box<Expr>>, Option<Box<Expr>>),
    Sub(Option<Box<Expr>>, Option<Box<Expr>>),
    Mul(Option<Box<Expr>>, Option<Box<Expr>>),
    Div(Option<Box<Expr>>, Option<Box<Expr>>),
    Exp(Option<Box<Expr>>, Option<Box<Expr>>),
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => {
                Value::Int(left + right)
            },
            (Value::Int(left), Value::Float(right)) => {
                Value::Float(left as f64 + right)
            },
            (Value::Float(left), Value::Int(right)) => {
                Value::Float(left + right as f64)
            },
            (Value::Float(left), Value::Float(right)) => {
                Value::Float(left + right)
            },
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => {
                Value::Int(left - right)
            },
            (Value::Int(left), Value::Float(right)) => {
                Value::Float(left as f64 - right)
            },
            (Value::Float(left), Value::Int(right)) => {
                Value::Float(left - right as f64)
            },
            (Value::Float(left), Value::Float(right)) => {
                Value::Float(left - right)
            },
        }
    }
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => {
                Value::Int(left * right)
            },
            (Value::Int(left), Value::Float(right)) => {
                Value::Float(left as f64 * right)
            },
            (Value::Float(left), Value::Int(right)) => {
                Value::Float(left * right as f64)
            },
            (Value::Float(left), Value::Float(right)) => {
                Value::Float(left * right)
            },
        }
    }
}

impl std::ops::Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => {
                Value::Int(left / right)
            },
            (Value::Int(left), Value::Float(right)) => {
                Value::Float(left as f64 / right)
            },
            (Value::Float(left), Value::Int(right)) => {
                Value::Float(left / right as f64)
            },
            (Value::Float(left), Value::Float(right)) => {
                Value::Float(left / right)
            },
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(x) => {
                f.write_str(&x.to_string())
            },
            Value::Float(x) => {
                f.write_str(&x.to_string())
            },
        }
    }
}

impl Value {
    fn exp(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => {
                Value::Int(left.pow(right as u32))
            },
            (Value::Int(left), Value::Float(right)) => {
                Value::Float((left as f64).powf(right))
            },
            (Value::Float(left), Value::Int(right)) => {
                Value::Float(left.powf(right as f64))
            },
            (Value::Float(left), Value::Float(right)) => {
                Value::Float(left.powf(right))
            },
            
        }
    }
}



impl Expr {
    pub fn from_tokens(tokens: Vec<tokenizer::Token>) -> Expr {
        todo!("Construct Syntax Tree from Tokens");
    }

    pub fn evaluate(&self) -> Result<Value, ()> {
        match self {
            Expr::Parens(sub_expr) => {
                if let Some(expr) = sub_expr {
                    expr.evaluate()
                } else {
                    Err(())
                }
            },
            Expr::Number(v) => {
                Ok(v.clone())
            },
            Expr::Add(left, right) => {
                let left = eval_or_err(left);
                let right = eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left + right)
                } else {
                    Err(())
                }
            },
            Expr::Sub(left, right) => {
                let left = eval_or_err(left);
                let right = eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left - right)
                } else {
                    Err(())
                }
            },
            Expr::Mul(left, right) => {
                let left = eval_or_err(left);
                let right = eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left * right)
                } else {
                    Err(())
                }
            },
            Expr::Div(left, right) => {
                let left = eval_or_err(left);
                let right = eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left / right)
                } else {
                    Err(())
                }
            },
            Expr::Exp(left, right) => {
                let left = eval_or_err(left);
                let right = eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left.exp(right))
                } else {
                    Err(())
                }
            },
        }
    }
}

fn eval_or_err(val: &Option<Box<Expr>>) -> Result<Value, ()> {
    if let Some(val) = val {
        val.evaluate()
    } else {
        Err(())
    }
}
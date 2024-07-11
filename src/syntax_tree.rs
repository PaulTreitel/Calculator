use std::fmt::Display;

use super::tokenizer;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub enum ExprTree {
    Parens(Option<Box<ExprTree>>),
    CloseParen,
    Number(Value),
    Add(Option<Box<ExprTree>>, Option<Box<ExprTree>>),
    Sub(Option<Box<ExprTree>>, Option<Box<ExprTree>>),
    Mul(Option<Box<ExprTree>>, Option<Box<ExprTree>>),
    Div(Option<Box<ExprTree>>, Option<Box<ExprTree>>),
    Exp(Option<Box<ExprTree>>, Option<Box<ExprTree>>),
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

    fn from_str(s: &str) -> Value {
        if s.contains("\\.") {
            Value::Float(s.parse::<f64>().ok().unwrap())
        } else {
            Value::Int(s.parse::<i64>().ok().unwrap())
        }
    }
}



impl ExprTree {
    pub fn from_tokens(tokens: &Vec<tokenizer::Token>) -> ExprTree {
        let mut expr_vec = Self::direct_token_convert(&tokens);
        Self::tree_from_exprs(&mut expr_vec)
        // todo!("Construct Syntax Tree from Tokens");
    }

    fn tree_from_exprs(exprs: &mut Vec<ExprTree>) -> ExprTree {
        // replaces open and closing parens with parenthesis trees
        // then follow order of operations
        Self::from_vec_parens(exprs);
        Self::from_vec_exp(exprs);
        Self::from_vec_mul_div(exprs);
        Self::from_vec_add_sub(exprs);

        ExprTree::CloseParen
    }

    fn from_vec_parens(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();
        for idx in 0..exprs.len() {
            let expr = exprs.get(idx).unwrap();
            match expr {
                ExprTree::Parens(_) => (),
                _ => {
                    continue
                },
            }
            let end_paren_idx = Self::get_closing_paren_idx(exprs, idx);
            let mut paren_subexpr = exprs[idx+1..end_paren_idx].as_mut().to_vec();
            let subexpr_tree = Self::tree_from_exprs(&mut paren_subexpr);
            changes.push((idx, ExprTree::Parens(Some(Box::new(subexpr_tree)))));
            // keep `expr` but delete the closing parenthesis
            remove_idxs.push((idx + 1, end_paren_idx + 1));
        }

        while !changes.is_empty() {
            let (idx, val) = changes.pop().unwrap();
            *exprs.get_mut(idx).unwrap() = val;
        }

        while !remove_idxs.is_empty() {
            let (start, end) = remove_idxs.pop().unwrap();
            for i in (start..end).rev() {
                exprs.remove(i);
            }
        }
    }

    fn get_closing_paren_idx(exprs: &Vec<ExprTree>, idx: usize) -> usize {
        let mut open_paren_ct = 0;
        for i in idx..exprs.len() {
            match exprs.get(i).unwrap() {
                ExprTree::Parens(_) => {
                    open_paren_ct += 1;
                },
                ExprTree::CloseParen => {
                    if open_paren_ct == 0 {
                        return i;
                    }
                }
                _ => (),
            }
        }
        panic!("Trying to find a closing parenthesis but there are none!")
    }

    fn from_vec_exp(exprs: &mut Vec<ExprTree>) -> () {
        todo!("Expressionize Exponents")
    }

    fn from_vec_mul_div(exprs: &mut Vec<ExprTree>) -> () {
        todo!("Expressionize Multiplication and Division")
    }

    fn from_vec_add_sub(exprs: &mut Vec<ExprTree>) -> () {
        todo!("Expressionize Addition and Subtraction")
    }

    fn direct_token_convert(tokens: &Vec<tokenizer::Token>) -> Vec<ExprTree> {
        let mut exprs = Vec::new();
        for token in tokens {
            match token {
                tokenizer::Token::Number(x) => {
                    exprs.push(ExprTree::Number(Value::from_str(x)));
                },
                tokenizer::Token::OpenParen(_) => {
                    exprs.push(ExprTree::Parens(None));
                },
                tokenizer::Token::CloseParen(_) => {
                    exprs.push(ExprTree::CloseParen);
                },
                tokenizer::Token::Add => {
                    exprs.push(ExprTree::Add(None, None));
                },
                tokenizer::Token::Sub => {
                    exprs.push(ExprTree::Sub(None, None));
                },
                tokenizer::Token::Mul => {
                    exprs.push(ExprTree::Mul(None, None));
                },
                tokenizer::Token::Div => {
                    exprs.push(ExprTree::Div(None, None));
                },
                tokenizer::Token::Exp => {
                    exprs.push(ExprTree::Exp(None, None));
                },
            }
        }
        exprs
    }

    pub fn evaluate(&self) -> Result<Value, ()> {
        match self {
            ExprTree::Parens(sub_expr) => {
                if let Some(expr) = sub_expr {
                    expr.evaluate()
                } else {
                    Err(())
                }
            },
            ExprTree::Number(v) => {
                Ok(v.clone())
            },
            ExprTree::Add(left, right) => {
                let left = Self::eval_or_err(left);
                let right = Self::eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left + right)
                } else {
                    Err(())
                }
            },
            ExprTree::Sub(left, right) => {
                let left = Self::eval_or_err(left);
                let right = Self::eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left - right)
                } else {
                    Err(())
                }
            },
            ExprTree::Mul(left, right) => {
                let left = Self::eval_or_err(left);
                let right = Self::eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left * right)
                } else {
                    Err(())
                }
            },
            ExprTree::Div(left, right) => {
                let left = Self::eval_or_err(left);
                let right = Self::eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left / right)
                } else {
                    Err(())
                }
            },
            ExprTree::Exp(left, right) => {
                let left = Self::eval_or_err(left);
                let right = Self::eval_or_err(right);
                if let (Ok(left), Ok(right)) = (left, right) {
                    Ok(left.exp(right))
                } else {
                    Err(())
                }
            },
            ExprTree::CloseParen => {
                Err(())
            },
        }
    }

    fn eval_or_err(val: &Option<Box<ExprTree>>) -> Result<Value, ()> {
        if let Some(val) = val {
            val.evaluate()
        } else {
            Err(())
        }
    }
}
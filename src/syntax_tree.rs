use std::fmt::Display;
use super::tokenizer::Token;

#[derive(Debug, Clone, PartialEq, Copy)]
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
                Value::Float(left as f64 / right as f64)
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
    pub fn from_tokens(tokens: &Vec<Token>) -> ExprTree {
        let mut expr_vec = Self::direct_token_convert(&tokens);
        Self::tree_from_exprs(&mut expr_vec)
    }

    fn tree_from_exprs(exprs: &mut Vec<ExprTree>) -> ExprTree {
        // replaces open and closing parens with parenthesis trees
        // then follow order of operations
        Self::from_vec_parens(exprs);
        Self::from_vec_exp(exprs);
        Self::from_vec_mul_div(exprs);
        Self::from_vec_add_sub(exprs);

        todo!("Expression the Tree")
    }

    fn from_vec_parens(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs: Vec<usize> = Vec::new();
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
            let rem: Vec<usize> = (idx + 1..end_paren_idx + 1).collect();
            remove_idxs.extend(rem);
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
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
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Exp(_,_) => {
                    remove_idxs.extend([idx - 1, idx + 1].iter());
                    changes.push((idx, ExprTree::Exp(
                        Some(Box::new(left.clone())),
                        Some(Box::new(right.clone())))));
                },
                _ => (),
            }
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn from_vec_mul_div(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Mul(_,_) => {
                    remove_idxs.extend([idx - 1, idx + 1].iter());
                    changes.push((idx, ExprTree::Mul(
                        Some(Box::new(left.clone())),
                        Some(Box::new(right.clone())))));
                },
                ExprTree::Div(_,_) => {
                    remove_idxs.extend([idx - 1, idx + 1].iter());
                    changes.push((idx, ExprTree::Div(
                        Some(Box::new(left.clone())),
                        Some(Box::new(right.clone())))));
                },
                _ => (),
            }
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn from_vec_add_sub(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Add(_,_) => {
                    remove_idxs.extend([idx - 1, idx + 1].iter());
                    changes.push((idx, ExprTree::Add(
                        Some(Box::new(left.clone())),
                        Some(Box::new(right.clone())))));
                },
                ExprTree::Div(_,_) => {
                    remove_idxs.extend([idx - 1, idx + 1].iter());
                    changes.push((idx, ExprTree::Add(
                        Some(Box::new(left.clone())),
                        Some(Box::new(right.clone())))));
                },
                _ => (),
            }
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn update_expr_vec(
        exprs: &mut Vec<ExprTree>, 
        changes: &mut Vec<(usize, ExprTree)>,
        remove_idxs: &mut Vec<usize>
    ) -> () {
        while !changes.is_empty() {
            let (idx, val) = changes.pop().unwrap();
            *exprs.get_mut(idx).unwrap() = val;
        }
        
        while !remove_idxs.is_empty() {
            exprs.remove(remove_idxs.pop().unwrap());
        }
    }

    fn direct_token_convert(tokens: &Vec<Token>) -> Vec<ExprTree> {
        let mut exprs = Vec::new();
        for token in tokens {
            match token {
                Token::Number(x) => {
                    exprs.push(ExprTree::Number(Value::from_str(x)));
                },
                Token::OpenParen(_) => {
                    exprs.push(ExprTree::Parens(None));
                },
                Token::CloseParen(_) => {
                    exprs.push(ExprTree::CloseParen);
                },
                Token::Add => {
                    exprs.push(ExprTree::Add(None, None));
                },
                Token::Sub => {
                    exprs.push(ExprTree::Sub(None, None));
                },
                Token::Mul => {
                    exprs.push(ExprTree::Mul(None, None));
                },
                Token::Div => {
                    exprs.push(ExprTree::Div(None, None));
                },
                Token::Exp => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_arithmetic() {
        let val1 = Value::Int(23);
        let val2 = Value::Int(4);
        let val3 = Value::Float(0.5);
        let val4 = Value::Float(2.2);

        assert_eq!(Value::Int(46), val1 + val1);
        assert_eq!(Value::Int(-19), val2 - val1);
        assert_eq!(Value::Int(529), val1 * val1);
        assert_eq!(Value::Float(1.0), val1 / val1);
        assert_eq!(Value::Float(5.75), val1 / val2);
        assert_eq!(Value::Int(279841), val1.exp(val2));

        assert_eq!(Value::Float(23.5), val1 + val3);
        assert_eq!(Value::Float(20.8), val1 - val4);
        assert_eq!(Value::Float(11.5), val1 * val3);
        assert_eq!(Value::Float(46.0), val1 / val3);
        assert_eq!(Value::Float(2.0), val2.exp(val3));
        assert_eq!(Value::Float(0.0625), val3.exp(val2));

        assert_eq!(Value::Float(2.7), val3 + val4);
        // chosen so as to avoid floating-point inequality problems
        assert_eq!(Value::Float(-1.1), Value::Float(1.1) - val4);
        assert_eq!(Value::Float(1.1), val3 * val4);
        assert_eq!(Value::Float(4.4), val4 / val3);
        assert_eq!(Value::Float(0.25), val3.exp(Value::Float(2.0)));
    }
}
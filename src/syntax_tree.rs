use std::fmt::Display;
use std::error::Error;
use super::tokenizer::Token;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Value {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub enum EvaluationError {
    EmptyParenthesisError,
    MissingAddOperand,
    MissingSubOperand,
    MissingMulOperand,
    MissingDivOperand,
    MissingExpOperand,
    ExtantCloseParen,
}

impl Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationError::EmptyParenthesisError => {
                f.write_str("a pair of parentheses contains nothing")
            },
            EvaluationError::MissingAddOperand => {
                f.write_str("an addition operator is missing an operand")
            },
            EvaluationError::MissingSubOperand => {
                f.write_str("a subtraction operator is missing an operand")
            },
            EvaluationError::MissingMulOperand => {
                f.write_str("a multiplication operator is missing an operand")
            },
            EvaluationError::MissingDivOperand => {
                f.write_str("a division operator is missing an operand")
            },
            EvaluationError::MissingExpOperand => {
                f.write_str("an exponentiation operator is missing an operand")
            },
            EvaluationError::ExtantCloseParen => {
                f.write_str("a closing parenthesis exists but shouldn't")
            },
        }
    }
}

impl Error for EvaluationError {

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
        let str = match self {
            Value::Float(x) => {
                x.to_string()
            }
            Value::Int(x) => {
                x.to_string()
            }
        };
        f.write_str(&str)
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

    fn from_string(s: &str) -> Value {
        if s.contains(".") {
            Value::Float(s.parse::<f64>().ok().unwrap())
        } else {
            Value::Int(s.parse::<i64>().ok().unwrap())
        }
    }
}

impl ExprTree {
    pub fn from_tokens(tokens: &Vec<Token>) -> ExprTree {
        let expr_vec = Self::direct_token_convert(&tokens);
        Self::tree_from_exprs(expr_vec)
    }

    fn tree_from_exprs(mut exprs: Vec<ExprTree>) -> ExprTree {
        // replaces open and closing parens with parenthesis trees
        // then follow order of operations
        Self::parens_tree_from_vec(&mut exprs);
        Self::exp_tree_from_vec(&mut exprs);
        Self::mul_div_tree_from_vec(&mut exprs);
        Self::add_sub_tree_from_vec(&mut exprs);
        
        if exprs.len() == 1 {
            exprs.pop().unwrap()
        } else {
            panic!("expression vector not 1 tree: {:?}", exprs);
        }
    }

    fn parens_tree_from_vec(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs: Vec<usize> = Vec::new();
        let mut changes = Vec::new();
        for idx in 0..exprs.len() {
            let expr = exprs.get(idx).unwrap();
            match expr {
                ExprTree::Parens(None) => (),
                _ => {
                    continue
                },
            }
            if remove_idxs.contains(&idx) {
                continue;
            }
            let end_paren_idx = Self::get_closing_paren_idx(exprs, idx);
            let paren_subexpr = exprs[idx+1..end_paren_idx].to_vec();
            let subexpr_tree = Self::tree_from_exprs(paren_subexpr);
            changes.push((idx, ExprTree::Parens(Some(Box::new(subexpr_tree)))));
            // keep `expr` but delete the closing parenthesis
            let rem: Vec<usize> = (idx + 1..end_paren_idx + 1).collect();
            remove_idxs.extend(rem);
        }
        let mut changes: Vec<(usize, ExprTree)> = changes.iter()
            .filter(|x| !remove_idxs.contains(&x.0))
            .map(|x|x.to_owned())
            .collect();
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn get_closing_paren_idx(exprs: &Vec<ExprTree>, idx: usize) -> usize {
        let mut open_paren_ct = 0;
        for i in idx + 1..exprs.len() {
            match exprs.get(i).unwrap() {
                ExprTree::Parens(_) => {
                    open_paren_ct += 1;
                },
                ExprTree::CloseParen => {
                    if open_paren_ct == 0 {
                        return i;
                    } else {
                        open_paren_ct -= 1;
                    }
                }
                _ => (),
            }
        }
        panic!("No closing parenthesis! open parenthesis at index {}", idx);
    }

    fn exp_tree_from_vec(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Exp(_,_) => {
                    if !remove_idxs.contains(&(idx - 1)) {
                        remove_idxs.extend([idx - 1, idx + 1].iter());
                        changes.push((
                            idx, 
                            ExprTree::Exp(
                                Some(Box::new(left.clone())),
                                Some(Box::new(right.clone())))));
                    } else {
                        remove_idxs.extend([idx, idx + 1].iter());
                        let new_left = changes.pop().unwrap().1;
                        changes.push((
                            idx - 2,
                            ExprTree::Exp(
                                Some(Box::new(new_left)), 
                                Some(Box::new(right.clone()))
                            )
                        ));
                    }
                },
                _ => (),
            }
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn mul_div_tree_from_vec(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Mul(_,_) => {
                    if !remove_idxs.contains(&(idx - 1)) {
                        remove_idxs.extend([idx - 1, idx + 1].iter());
                        changes.push((
                            idx, 
                            ExprTree::Mul(
                                Some(Box::new(left.clone())),
                                Some(Box::new(right.clone())))));
                    } else {
                        remove_idxs.extend([idx, idx + 1].iter());
                        let new_left = changes.pop().unwrap().1;
                        changes.push((
                            idx - 2,
                            ExprTree::Mul(
                                Some(Box::new(new_left)), 
                                Some(Box::new(right.clone()))
                            )
                        ));
                    }
                },
                ExprTree::Div(_,_) => {
                    if !remove_idxs.contains(&(idx - 1)) {
                        remove_idxs.extend([idx - 1, idx + 1].iter());
                        changes.push((
                            idx, 
                            ExprTree::Div(
                                Some(Box::new(left.clone())),
                                Some(Box::new(right.clone())))));
                    } else {
                        remove_idxs.extend([idx, idx + 1].iter());
                        let new_left = changes.pop().unwrap().1;
                        changes.push((
                            idx - 2,
                            ExprTree::Div(
                                Some(Box::new(new_left)), 
                                Some(Box::new(right.clone()))
                            )
                        ));
                    }
                },
                _ => (),
            }
        }
        Self::update_expr_vec(exprs, &mut changes, &mut remove_idxs);
    }

    fn add_sub_tree_from_vec(exprs: &mut Vec<ExprTree>) -> () {
        let mut remove_idxs = Vec::new();
        let mut changes = Vec::new();

        for idx in 1..exprs.len() - 1 {
            let expr = exprs.get(idx).unwrap();
            let left = exprs.get(idx - 1).unwrap();
            let right = exprs.get(idx + 1).unwrap();
            match expr {
                ExprTree::Add(_,_) => {
                    if !remove_idxs.contains(&(idx - 1)) {
                        remove_idxs.extend([idx - 1, idx + 1].iter());
                        changes.push((
                            idx, 
                            ExprTree::Add(
                                Some(Box::new(left.clone())),
                                Some(Box::new(right.clone())))));
                    } else {
                        remove_idxs.extend([idx, idx + 1].iter());
                        let new_left = changes.pop().unwrap().1;
                        changes.push((
                            idx - 2,
                            ExprTree::Add(
                                Some(Box::new(new_left)), 
                                Some(Box::new(right.clone()))
                            )
                        ));
                    }
                },
                ExprTree::Sub(_,_) => {
                    if !remove_idxs.contains(&(idx - 1)) {
                        remove_idxs.extend([idx - 1, idx + 1].iter());
                        changes.push((
                            idx, 
                            ExprTree::Sub(
                                Some(Box::new(left.clone())),
                                Some(Box::new(right.clone())))));
                    } else {
                        remove_idxs.extend([idx, idx + 1].iter());
                        let new_left = changes.pop().unwrap().1;
                        changes.push((
                            idx - 2,
                            ExprTree::Sub(
                                Some(Box::new(new_left)), 
                                Some(Box::new(right.clone()))
                            )
                        ));
                    }
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
                    exprs.push(ExprTree::Number(Value::from_string(x)));
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

    pub fn evaluate(&self) -> Result<Value, EvaluationError> {
        match self {
            ExprTree::Parens(sub_expr) => {
                if let Some(expr) = sub_expr {
                    expr.evaluate()
                } else {
                    Err(EvaluationError::EmptyParenthesisError)
                }
            },
            ExprTree::Number(v) => {
                Ok(v.clone())
            },
            ExprTree::Add(left, right) => {
                if left.is_none() || right.is_none() {
                    return Err(EvaluationError::MissingAddOperand);
                }
                let left = left.clone().unwrap().evaluate();
                let right = right.clone().unwrap().evaluate();
                match (left, right) {
                    (Ok(left), Ok(right)) => {
                        Ok(left + right)
                    },
                    (Ok(_), Err(e)) => {
                        Err(e)
                    },
                    (Err(e), Ok(_)) => {
                        Err(e)
                    },
                    (Err(e1), Err(_)) => {
                        Err(e1)
                    },
                }
            },
            ExprTree::Sub(left, right) => {
                if left.is_none() || right.is_none() {
                    return Err(EvaluationError::MissingSubOperand);
                }
                let left = left.clone().unwrap().evaluate();
                let right = right.clone().unwrap().evaluate();
                match (left, right) {
                    (Ok(left), Ok(right)) => {
                        Ok(left - right)
                    },
                    (Ok(_), Err(e)) => {
                        Err(e)
                    },
                    (Err(e), Ok(_)) => {
                        Err(e)
                    },
                    (Err(e1), Err(_)) => {
                        Err(e1)
                    },
                }
            },
            ExprTree::Mul(left, right) => {
                if left.is_none() || right.is_none() {
                    return Err(EvaluationError::MissingMulOperand);
                }
                let left = left.clone().unwrap().evaluate();
                let right = right.clone().unwrap().evaluate();
                match (left, right) {
                    (Ok(left), Ok(right)) => {
                        Ok(left * right)
                    },
                    (Ok(_), Err(e)) => {
                        Err(e)
                    },
                    (Err(e), Ok(_)) => {
                        Err(e)
                    },
                    (Err(e1), Err(_)) => {
                        Err(e1)
                    },
                }
            },
            ExprTree::Div(left, right) => {
                if left.is_none() || right.is_none() {
                    return Err(EvaluationError::MissingDivOperand);
                }
                let left = left.clone().unwrap().evaluate();
                let right = right.clone().unwrap().evaluate();
                match (left, right) {
                    (Ok(left), Ok(right)) => {
                        Ok(left / right)
                    },
                    (Ok(_), Err(e)) => {
                        Err(e)
                    },
                    (Err(e), Ok(_)) => {
                        Err(e)
                    },
                    (Err(e1), Err(_)) => {
                        Err(e1)
                    },
                }
            },
            ExprTree::Exp(left, right) => {
                if left.is_none() || right.is_none() {
                    return Err(EvaluationError::MissingExpOperand);
                }
                let left = left.clone().unwrap().evaluate();
                let right = right.clone().unwrap().evaluate();
                match (left, right) {
                    (Ok(left), Ok(right)) => {
                        Ok(left.exp(right))
                    },
                    (Ok(_), Err(e)) => {
                        Err(e)
                    },
                    (Err(e), Ok(_)) => {
                        Err(e)
                    },
                    (Err(e1), Err(_)) => {
                        Err(e1)
                    },
                }
            },
            ExprTree::CloseParen => {
                Err(EvaluationError::ExtantCloseParen)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_full_expr_vec() -> Vec<ExprTree> {
        vec![
            ExprTree::Number(Value::Int(3)),
            ExprTree::Add(None, None),
            ExprTree::Parens(None),
            ExprTree::Number(Value::Int(5)),
            ExprTree::Mul(None, None),
            ExprTree::Number(Value::Float(2.2)),
            ExprTree::Exp(None, None),
            ExprTree::Number(Value::Float(0.4)),
            ExprTree::CloseParen,
            ExprTree::Sub(None, None),
            ExprTree::Number(Value::Int(6)),
            ExprTree::Div(None, None),
            ExprTree::Number(Value::Float(1.5)),
        ]
    }

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

    #[test]
    fn value_from_str() {
        assert_eq!(Value::Int(33), Value::from_string("33"));
        assert_eq!(Value::Float(33.0), Value::from_string("33."));
        assert_eq!(Value::Float(33.0), Value::from_string("33.0"));
        assert_eq!(Value::Float(0.4), Value::from_string("0.4"));
        assert_eq!(Value::Float(0.4), Value::from_string(".4"));
        assert_eq!(Value::Float(0.4), Value::from_string(".40"));
    }

    #[test]
    fn updating_exprs() {
        let mut val1 = get_full_expr_vec();
        let len1 = val1.len();
        let mut changes: Vec<(usize, ExprTree)> = vec![
            (1, ExprTree::Add(
                Some(Box::new(ExprTree::Number(Value::Int(3)))), 
                None)),
            (6, ExprTree::Exp(
                Some(Box::new(ExprTree::Number(Value::Float(2.2)))),
                Some(Box::new(ExprTree::Number(Value::Float(0.4))))))
        ];
        let out_changes: Vec<ExprTree> = changes.iter().map(|x|x.1.clone()).collect();
        let mut remove_idxs: Vec<usize> = vec![0, 5, 7];
        ExprTree::update_expr_vec(&mut val1, &mut changes, &mut remove_idxs);

        assert!(changes.is_empty());
        assert!(remove_idxs.is_empty());
        assert_eq!(val1.len(), len1 - 3);
        assert_eq!(val1.get(0).unwrap(), out_changes.get(0).unwrap());
        assert_eq!(val1.get(4).unwrap(), out_changes.get(1).unwrap());
        assert!(!val1.contains(&ExprTree::Number(Value::Int(3))));
        assert!(!val1.contains(&ExprTree::Number(Value::Float(2.2))));
        assert!(!val1.contains(&ExprTree::Number(Value::Float(0.4))));
    }

    #[test]
    fn convert_tokens() {
        let tokens = vec![
            Token::Number("3"),
            Token::Add,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Mul,
            Token::Number("2.2"),
            Token::Exp,
            Token::Number(".4"),
            Token::CloseParen("]"),
            Token::Sub,
            Token::Number("6"),
            Token::Div,
            Token::Number("1.5"),
        ];
        let val1 = get_full_expr_vec();
        let new_expr_vec = ExprTree::direct_token_convert(&tokens);
        assert_eq!(val1, new_expr_vec);
    }

    #[test]
    fn combine_add_sub() {
        let mut val1 = get_full_expr_vec();
        let len1 = val1.len();
        ExprTree::add_sub_tree_from_vec(&mut val1);
        assert_eq!(val1.len(), len1 - 4);
        assert_eq!(
            val1.get(0).unwrap(),
            &ExprTree::Add(
                Some(Box::new(ExprTree::Number(Value::Int(3)))),
                Some(Box::new(ExprTree::Parens(None)))
            )
        );
        assert_eq!(
            val1.get(6).unwrap(),
            &ExprTree::Sub(
                Some(Box::new(ExprTree::CloseParen)),
                Some(Box::new(ExprTree::Number(Value::Int(6))))
            )
        );
    }

    #[test]
    fn combine_mul_div() {
        let mut val1 = get_full_expr_vec();
        let len1 = val1.len();
        ExprTree::mul_div_tree_from_vec(&mut val1);
        assert_eq!(val1.len(), len1 - 4);
        assert_eq!(
            val1.get(3).unwrap(),
            &ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Int(5)))),
                Some(Box::new(ExprTree::Number(Value::Float(2.2))))
            )
        );
        assert_eq!(
            val1.get(8).unwrap(),
            &ExprTree::Div(
                Some(Box::new(ExprTree::Number(Value::Int(6)))),
                Some(Box::new(ExprTree::Number(Value::Float(1.5))))
            )
        );
    }

    #[test]
    fn combine_exp() {
        let mut val1 = get_full_expr_vec();
        let len1 = val1.len();
        ExprTree::exp_tree_from_vec(&mut val1);
        assert_eq!(val1.len(), len1 - 2);
        assert_eq!(
            val1.get(5).unwrap(),
            &ExprTree::Exp(
                Some(Box::new(ExprTree::Number(Value::Float(2.2)))),
                Some(Box::new(ExprTree::Number(Value::Float(0.4))))
            )
        );
    }

    #[test]
    fn find_closing_paren() {
        let val1 = get_full_expr_vec();
        assert_eq!(ExprTree::get_closing_paren_idx(&val1, 2), 8);
        assert_eq!(ExprTree::get_closing_paren_idx(&val1, 5), 8);

        let mut val2 = get_full_expr_vec();
        val2.extend(val2.clone());
        assert_eq!(ExprTree::get_closing_paren_idx(&val2, 15), 21);
    }

    #[test]
    #[should_panic]
    fn find_closing_paren_none() {
        let val1 = get_full_expr_vec();
        ExprTree::get_closing_paren_idx(&val1, 9);
    }

    #[test]
    #[should_panic]
    fn find_closing_paren_extra_opening_paren() {
        let val1 = get_full_expr_vec();
        ExprTree::get_closing_paren_idx(&val1, 0);
    }

    #[test]
    fn combine_parens_simple() {
        let mut val1 = vec![
            ExprTree::Number(Value::Int(3)),
            ExprTree::Add(None, None),
            ExprTree::Parens(None),
            ExprTree::Number(Value::Int(5)),
            ExprTree::CloseParen,
        ];
        ExprTree::parens_tree_from_vec(&mut val1);
        assert_eq!(val1.len(), 3);
        assert_eq!(val1.get(0).unwrap(), &ExprTree::Number(Value::Int(3)));
        assert_eq!(val1.get(1).unwrap(), &ExprTree::Add(None, None));
        assert_eq!(
            val1.get(2).unwrap(), 
            &ExprTree::Parens(
                Some(Box::new(ExprTree::Number(Value::Int(5))))
            )
        );
    }

    #[test]
    #[should_panic]
    fn combine_parens_empty() {
        let mut val1 = vec![
            ExprTree::Number(Value::Int(3)),
            ExprTree::Add(None, None),
            ExprTree::Parens(None),
            ExprTree::CloseParen,
        ];
        ExprTree::parens_tree_from_vec(&mut val1);
    }

    #[test]
    fn combine_parens_complex() {
        let mut val1 = get_full_expr_vec();
        let expected = vec![
            ExprTree::Number(Value::Int(3)),
            ExprTree::Add(None, None),
            ExprTree::Parens(
                Some(Box::new(ExprTree::Mul(
                    Some(Box::new(ExprTree::Number(Value::Int(5)))),
                    Some(Box::new(ExprTree::Exp(
                        Some(Box::new(ExprTree::Number(Value::Float(2.2)))), 
                        Some(Box::new(ExprTree::Number(Value::Float(0.4))))
                    )))
                )))
            ),
            ExprTree::Sub(None, None),
            ExprTree::Number(Value::Int(6)),
            ExprTree::Div(None, None),
            ExprTree::Number(Value::Float(1.5)),
        ];
        ExprTree::parens_tree_from_vec(&mut val1);
        assert_eq!(val1.len(), 7);
        assert_eq!(val1, expected);
    }

    #[test]
    fn complete_tree() {
        let val1 = get_full_expr_vec();
        let expected = ExprTree::Sub(
            Some(Box::new(
                ExprTree::Add(
                    Some(Box::new(ExprTree::Number(Value::Int(3)))),
                    Some(Box::new(
                        ExprTree::Parens(
                            Some(Box::new(ExprTree::Mul(
                                Some(Box::new(ExprTree::Number(Value::Int(5)))),
                                Some(Box::new(ExprTree::Exp(
                                    Some(Box::new(ExprTree::Number(Value::Float(2.2)))), 
                                    Some(Box::new(ExprTree::Number(Value::Float(0.4))))
                                )))
                            )))
                        ),
                    ))
                ),
            )),
            Some(Box::new(
                ExprTree::Div(
                    Some(Box::new(ExprTree::Number(Value::Int(6)))),
                    Some(Box::new(ExprTree::Number(Value::Float(1.5))))
                ),
            ))
        );
        let tree = ExprTree::tree_from_exprs(val1);
        assert_eq!(tree, expected);
    }

    #[test]
    fn full_evaluation() {
        let val1 = vec![
            ExprTree::Number(Value::Int(3)),
            ExprTree::Add(None, None),
            ExprTree::Parens(None),
            ExprTree::Number(Value::Int(5)),
            ExprTree::Mul(None, None),
            ExprTree::Number(Value::Int(8)),
            ExprTree::Exp(None, None),
            ExprTree::Parens(None),
            ExprTree::Number(Value::Int(1)),
            ExprTree::Div(None, None),
            ExprTree::Number(Value::Int(3)),
            ExprTree::CloseParen,
            ExprTree::CloseParen,
            ExprTree::Sub(None, None),
            ExprTree::Number(Value::Int(6)),
            ExprTree::Div(None, None),
            ExprTree::Number(Value::Float(1.5)),
        ];
        let tree1 = ExprTree::tree_from_exprs(val1);
        let result1 = tree1.evaluate();
        assert!(result1.is_ok());
        assert_eq!(result1.ok().unwrap(), Value::Float(9.0));

        let val2 = vec![
            ExprTree::Parens(None),
            ExprTree::Number(Value::Float(3.5)),
            ExprTree::Exp(None, None),
            ExprTree::Number(Value::Int(2)),
            ExprTree::Sub(None, None),
            ExprTree::Number(Value::Int(3)),
            ExprTree::CloseParen,
            ExprTree::Mul(None, None),
            ExprTree::Number(Value::Int(5))
        ];
        let tree2 = ExprTree::tree_from_exprs(val2);
        let result2 = tree2.evaluate();
        assert!(result2.is_ok());
        assert_eq!(result2.ok().unwrap(), Value::Float(46.25));
    }

    #[test]
    fn eval_empty_paren() {
        let val1 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                Some(Box::new(ExprTree::Parens(None)))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::EmptyParenthesisError));
    }

    #[test]
    fn eval_close_paren() {
        let val1 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                Some(Box::new(ExprTree::CloseParen))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::ExtantCloseParen));
    }

    #[test]
    fn eval_missing_add_operand() {
        let val1 = ExprTree::Add(
            None, 
            Some(Box::new(ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                Some(Box::new(ExprTree::Number(Value::Int(10))))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::MissingAddOperand));

        let val2 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            None
        );
        let result2 = val2.evaluate();
        assert_eq!(result2, Err(EvaluationError::MissingAddOperand));
    }

    #[test]
    fn eval_missing_sub_operand() {
        let val1 = ExprTree::Sub(
            None, 
            Some(Box::new(ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                Some(Box::new(ExprTree::Number(Value::Int(10))))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::MissingSubOperand));

        let val2 = ExprTree::Sub(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            None
        );
        let result2 = val2.evaluate();
        assert_eq!(result2, Err(EvaluationError::MissingSubOperand));
    }

    #[test]
    fn eval_missing_mul_operand() {
        let val1 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Mul(
                None, 
                Some(Box::new(ExprTree::Number(Value::Int(10))))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::MissingMulOperand));

        let val2 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Mul(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                None
            )))
        );
        let result2 = val2.evaluate();
        assert_eq!(result2, Err(EvaluationError::MissingMulOperand));
    }

    #[test]
    fn eval_missing_div_operand() {
        let val1 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Div(
                None, 
                Some(Box::new(ExprTree::Number(Value::Int(10))))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::MissingDivOperand));

        let val2 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Div(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                None
            )))
        );
        let result2 = val2.evaluate();
        assert_eq!(result2, Err(EvaluationError::MissingDivOperand));
    }

    #[test]
    fn eval_missing_exp_operand() {
        let val1 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Exp(
                None, 
                Some(Box::new(ExprTree::Number(Value::Int(10))))
            )))
        );
        let result1 = val1.evaluate();
        assert_eq!(result1, Err(EvaluationError::MissingExpOperand));

        let val2 = ExprTree::Add(
            Some(Box::new(ExprTree::Number(Value::Int(3)))), 
            Some(Box::new(ExprTree::Exp(
                Some(Box::new(ExprTree::Number(Value::Float(5.75)))), 
                None
            )))
        );
        let result2 = val2.evaluate();
        assert_eq!(result2, Err(EvaluationError::MissingExpOperand));
    }
}
use logos::Logos;
use std::{error::Error, fmt::Display};

const OPERATORS: [Token<'_>; 5] = [Token::Add, Token::Sub, Token::Mul, Token::Div, Token::Exp];

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")]
pub enum Token<'a> {
    #[regex("(\\d*\\.)?\\d+")]
    Number(&'a str),

    #[regex("\\(|\\{|\\[")]
    OpenParen(&'a str),

    #[regex("\\)|\\]|\\}")]
    CloseParen(&'a str),

    #[token("+")]
    Add,

    #[token("-")]
    Sub,

    #[token("*")]
    Mul,

    #[token("/")]
    Div,

    #[token("^")]
    Exp,
}

#[derive(Debug, PartialEq)]
pub enum TokenValidationError<'a> {
    UnmatchedParens(usize),
    EmptyParens(usize),
    OperatorWithoutOperands(Token<'a>, usize),
    InvalidToken,
}

impl<'a> Token<'a> {
    fn is_operator(&self) -> bool {
        OPERATORS.contains(self)
    }
}

impl Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Token::Number(s) => s,
            Token::OpenParen(s) => s,
            Token::CloseParen(s) => s,
            Token::Add => "+",
            Token::Sub => "-",
            Token::Mul => "*",
            Token::Div => "/",
            Token::Exp => "^",
        };
        f.write_str(msg)
    }
}

impl Display for TokenValidationError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg;
        match self {
            TokenValidationError::UnmatchedParens(idx) => {
                msg = format!(
                    "there are unmatched or out-of-order parentheses at position {}",
                    idx + 1
                );
            },
            TokenValidationError::EmptyParens(idx) => {
                msg = format!(
                    "there are parentheses at position {} that are empty",
                    idx + 1
                );
            },
            TokenValidationError::OperatorWithoutOperands(t, idx) => {
                msg = format!(
                    "there is a {} at position {} that is missing an operand", 
                    t, 
                    idx + 1
                );
            },
            TokenValidationError::InvalidToken => {
                msg = format!("invalid character");
            },
        }
        f.write_str(msg.as_str())
    }
}

impl Error for TokenValidationError<'_> {
    
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, TokenValidationError> {
    let lex = Token::lexer(input);
    let mut all: Vec<Result<Token, ()>> = lex.collect();
    for idx in 0..all.len() {
        if all.get(idx).unwrap().clone().ok().is_none() {
            return Err(TokenValidationError::InvalidToken);
        }
    }
    let all: Vec<Token> = all.iter_mut()
        .map(|x| x.clone().ok().unwrap())
        .collect();
    Ok(all)
}

pub fn is_valid_expr<'a>(tokens: &'a Vec<Token<'a>>) -> Result<(), TokenValidationError<'a>> {
    if let Some(idx) = has_unmatched_parens(tokens) {
        return Err(TokenValidationError::UnmatchedParens(idx));
    }
    if let Some(idx) = has_empty_parens(tokens) {
        return Err(TokenValidationError::EmptyParens(idx));
    }
    if let Some((t, idx)) = has_start_end_ops(tokens) {
        return Err(TokenValidationError::OperatorWithoutOperands(t, idx));
    }
    if let Some((t, idx)) = has_ops_without_operands(tokens) {
        return Err(TokenValidationError::OperatorWithoutOperands(t, idx));
    }
    Ok(())
}

fn has_unmatched_parens(tokens: &Vec<Token>) -> Option<usize> {
    let mut open_parens = Vec::new();
    // for token in tokens {
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        match token {
            Token::OpenParen(open) => {
                open_parens.push((idx, open));
            },
            Token::CloseParen(close) => {
                let open = open_parens.pop();
                if open.is_none() {
                    return Some(idx);
                }
                if !open_close_paren_match(open.unwrap().1, close) {
                    return Some(idx);
                }
            },
            _ => (),
        }
    }
    if let Some((idx, _)) = open_parens.get(0) {
        Some(*idx)
    } else {
        None
    }
}

fn open_close_paren_match(open: &str, close: &str) -> bool {
    match open {
        "(" => close.eq(")"),
        "[" => close.eq("]"),
        "{" => close.eq("}"),
        _ => false
    }
}

fn has_empty_parens(tokens: &Vec<Token>) -> Option<usize> {
    let mut at_start = false;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            Token::CloseParen(_) => {
                if at_start {
                    return Some(idx - 1);
                }
                at_start = false;
            }
            _ => {
                at_start = false;
            },
        }
    }
    None
}

fn has_start_end_ops<'a>(tokens: &'a Vec<Token<'a>>) -> Option<(Token<'a>, usize)> {
    let mut at_start = true;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        if at_start && token.is_operator() && *token != Token::Sub {
            return Some((token.clone(), idx));
        }
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            Token::CloseParen(_) => {
                at_start = false;
                if let Some(t) = tokens.get(idx - 1) {
                    if t.is_operator() {
                        return Some((t.clone(), idx - 1));
                    }
                }
            }
            _ => {
                at_start = false;
            },
        }
    }
    let token = tokens.get(tokens.len() - 1).unwrap();
    match token.is_operator() {
        true => Some((token.clone(), tokens.len() - 1)),
        false => None
    }
}

fn has_ops_without_operands<'a>(tokens: &'a Vec<Token<'a>>) -> Option<(Token<'a>, usize)> {
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        if !token.is_operator() {
            continue;
        }
        // guaranteed these will exist because this is validated after has_start_end_ops
        let prev = tokens.get(idx - 1).unwrap();
        let next = tokens.get(idx + 1).unwrap();
        match prev {
            Token::CloseParen(_) => (),
            Token::Number(_) => (),
            _ => {
                return Some((token.clone(), idx));
            },
        }
        match next {
            Token::OpenParen(_) => (),
            Token::Number(_) => (),
            _ => {
                return Some((token.clone(), idx));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenization() {
        let input = "3+(5*2.2^.4]}{";
        let tokens = tokenize(input).ok().unwrap();
        assert_eq!(tokens, vec![
            Token::Number("3"),
            Token::Add,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Mul,
            Token::Number("2.2"),
            Token::Exp,
            Token::Number(".4"),
            Token::CloseParen("]"),
            Token::CloseParen("}"),
            Token::OpenParen("{"),
        ]);
    }

    #[test]
    fn open_period() {
        let input = "3+.";
        let tokens = tokenize(input);
        assert_eq!(tokens, Err(TokenValidationError::InvalidToken));
    }

    #[test]
    fn letters() {
        let input = "3a+2";
        let tokens = tokenize(input);
        assert_eq!(tokens, Err(TokenValidationError::InvalidToken));
    }

    #[test]
    fn matching_parens() {
        let val1 = vec![
            Token::OpenParen("("),
            Token::OpenParen("["),
            Token::OpenParen("{"),
            Token::CloseParen("}"),
            Token::CloseParen("]"),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_unmatched_parens(&val1), None);
    }

    #[test]
    fn unmatched_parens() {
        let val1 = vec![
            Token::OpenParen("("),
            Token::OpenParen("["),
            Token::CloseParen(")"),
            Token::CloseParen("]"),
        ];
        assert_eq!(has_unmatched_parens(&val1), Some(2));

        let val2 = vec![Token::OpenParen("(")];
        assert_eq!(has_unmatched_parens(&val2), Some(0));

        let val3 = vec![
            Token::CloseParen(")"),
            Token::OpenParen("("),
        ];
        assert_eq!(has_unmatched_parens(&val3), Some(0));

        let val4 = vec![Token::CloseParen(")")];
        assert_eq!(has_unmatched_parens(&val4), Some(0));
    }

    #[test]
    fn no_start_end_ops() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Number("3"),
        ];
        assert_eq!(has_start_end_ops(&val1), None);

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
            Token::Div,
            Token::Number("7"),
        ];
        assert_eq!(has_start_end_ops(&val2), None);
    }

    #[test]
    fn has_start_op() {
        let val1 = vec![
            Token::Add,
            Token::Number("3"),
        ];
        assert_eq!(has_start_end_ops(&val1), Some((Token::Add, 0)));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_start_end_ops(&val2), Some((Token::Add, 3)));
    }

    #[test]
    fn has_end_op() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
        ];
        assert_eq!(has_start_end_ops(&val1), Some((Token::Add, 1)));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Add,
            Token::CloseParen(")"),
            Token::Div,
            Token::Number("7"),
        ];
        assert_eq!(has_start_end_ops(&val2), Some((Token::Add, 4)));
    }

    #[test]
    fn no_empty_parens() {
        let val1 = vec![
            Token::OpenParen("("),
            Token::Number("2"),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_empty_parens(&val1), None);

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_empty_parens(&val2), None);
    }

    #[test]
    fn empty_parens() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::OpenParen("("),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_empty_parens(&val1), Some(2));

        let val2 = vec![
            Token::OpenParen("("),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_empty_parens(&val2), Some(0));

        let val3 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::CloseParen(")"),
            Token::Div,
            Token::Number("7"),
        ];
        assert_eq!(has_empty_parens(&val3), Some(2));
    }

    #[test]
    fn no_ops_without_operands() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Number("3"),
        ];
        assert_eq!(has_ops_without_operands(&val1), None);

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
            Token::Div,
            Token::Number("7"),
        ];
        assert_eq!(has_ops_without_operands(&val2), None);
    }

    #[test]
    fn ops_without_operands() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Add,
            Token::Number("3"),
        ];
        assert_eq!(has_ops_without_operands(&val1), Some((Token::Add, 1)));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
        ];
        assert_eq!(has_ops_without_operands(&val2), Some((Token::Add, 3)));

        let val3 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Add,
            Token::CloseParen(")"),
        ];
        assert_eq!(has_ops_without_operands(&val3), Some((Token::Add, 4)));
    }

    #[test]
    #[should_panic]
    fn ops_without_operands_with_start_end() {
        let val4 = vec![
            Token::Add,
            Token::Add,
            Token::Add,
        ];
        assert_eq!(has_ops_without_operands(&val4), Some((Token::Add, 0)));
    }
}
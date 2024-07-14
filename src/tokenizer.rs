use logos::Logos;

const OPERATORS: [Token<'_>; 5] = [Token::Add, Token::Sub, Token::Mul, Token::Div, Token::Exp];

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")]
pub enum Token<'a> {
    #[regex("[+-]?(\\d*\\.)?\\d+")]
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

impl<'a> Token<'a> {
    fn is_operator(&self) -> bool {
        OPERATORS.contains(self)
    }
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, ()> {
    let lex = Token::lexer(input);
    let mut all: Vec<Result<Token, ()>> = lex.collect();
    println!("{:?}", all);
    for idx in 0..all.len() {
        if all.get(idx).unwrap().clone().ok().is_none() {
            return Err(());
        }
    }
    let all: Vec<Token> = all.iter_mut()
        .map(|x| x.clone().ok().unwrap())
        .collect();
    Ok(all)
}

pub fn is_valid_expr(tokens: &Vec<Token>) -> bool {
    if has_unmatched_parens(tokens) {
        return false;
    }
    if has_empty_parens(tokens) {
        return false;
    }
    if has_start_end_ops(tokens) {
        return false;
    }
    if has_ops_without_operands(tokens) {
        return false;
    }
    true
}

fn has_unmatched_parens(tokens: &Vec<Token>) -> bool {
    let mut open_parens = Vec::new();
    for token in tokens {
        match token {
            Token::OpenParen(open) => {
                open_parens.push(open);
            },
            Token::CloseParen(close) => {
                let open = open_parens.pop();
                if open.is_none() {
                    return true;
                }
                if !open_close_paren_match(open.unwrap(), close) {
                    return true;
                }
            },
            _ => (),
        }
    }
    !open_parens.is_empty()
}

fn open_close_paren_match(open: &str, close: &str) -> bool {
    match open {
        "(" => close.eq(")"),
        "[" => close.eq("]"),
        "{" => close.eq("}"),
        _ => false
    }
}

fn has_empty_parens(tokens: &Vec<Token>) -> bool {
    let mut at_start = false;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            Token::CloseParen(_) => {
                if at_start {
                    return true;
                }
                at_start = false;
            }
            _ => {
                at_start = false;
            },
        }
    }
    false
}

fn has_start_end_ops(tokens: &Vec<Token>) -> bool {
    let mut at_start = true;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        if at_start && token.is_operator() {
            return true;
        }
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            Token::CloseParen(_) => {
                at_start = false;
                if let Some(t) = tokens.get(idx - 1) {
                    if t.is_operator() {
                        return true;
                    }
                }
            }
            _ => {
                at_start = false;
            },
        }
    }
    let token = tokens.get(tokens.len() - 1).unwrap();
    token.is_operator()
}

fn has_ops_without_operands(tokens: &Vec<Token>) -> bool {
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
                return true;
            },
        }
        match next {
            Token::OpenParen(_) => (),
            Token::Number(_) => (),
            _ => {
                return true;
            }
        }
    }
    false
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
        assert_eq!(tokens, Err(()));
    }

    #[test]
    fn letters() {
        let input = "3a+2";
        let tokens = tokenize(input);
        assert_eq!(tokens, Err(()));
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
        assert!(!has_unmatched_parens(&val1));
    }

    #[test]
    fn unmatched_parens() {
        let val1 = vec![
            Token::OpenParen("("),
            Token::OpenParen("["),
            Token::CloseParen(")"),
            Token::CloseParen("]"),
        ];
        assert!(has_unmatched_parens(&val1));

        let val2 = vec![Token::OpenParen("(")];
        assert!(has_unmatched_parens(&val2));

        let val3 = vec![
            Token::CloseParen(")"),
            Token::OpenParen("("),
        ];
        assert!(has_unmatched_parens(&val3));

        let val4 = vec![Token::CloseParen(")")];
        assert!(has_unmatched_parens(&val4));
    }

    #[test]
    fn no_start_end_ops() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Number("3"),
        ];
        assert!(!has_start_end_ops(&val1));

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
        assert!(!has_start_end_ops(&val2));
    }

    #[test]
    fn has_start_op() {
        let val1 = vec![
            Token::Add,
            Token::Number("3"),
        ];
        assert!(has_start_end_ops(&val1));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
        ];
        assert!(has_start_end_ops(&val2));
    }

    #[test]
    fn has_end_op() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
        ];
        assert!(has_start_end_ops(&val1));

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
        assert!(has_start_end_ops(&val2));
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
        assert!(!has_empty_parens(&val1));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::CloseParen(")"),
        ];
        assert!(!has_empty_parens(&val2));
    }

    #[test]
    fn empty_parens() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::OpenParen("("),
            Token::CloseParen(")"),
        ];
        assert!(has_empty_parens(&val1));

        let val2 = vec![
            Token::OpenParen("("),
            Token::CloseParen(")"),
        ];
        assert!(has_empty_parens(&val2));

        let val3 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::CloseParen(")"),
            Token::Div,
            Token::Number("7"),
        ];
        assert!(has_empty_parens(&val3));
    }

    #[test]
    fn no_ops_without_operands() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Number("3"),
        ];
        assert!(!has_ops_without_operands(&val1));

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
        assert!(!has_ops_without_operands(&val2));
    }

    #[test]
    fn ops_without_operands() {
        let val1 = vec![
            Token::Number("2"),
            Token::Add,
            Token::Add,
            Token::Number("3"),
        ];
        assert!(has_ops_without_operands(&val1));

        let val2 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Add,
            Token::Number("3"),
            Token::CloseParen(")"),
        ];
        assert!(has_ops_without_operands(&val2));

        let val3 = vec![
            Token::Number("2"),
            Token::Mul,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Add,
            Token::CloseParen(")"),
        ];
        assert!(has_ops_without_operands(&val3));
    }

    #[test]
    #[should_panic]
    fn ops_without_operands_with_start_end() {
        let val4 = vec![
            Token::Add,
            Token::Add,
            Token::Add,
        ];
        assert!(has_ops_without_operands(&val4));
    }
}
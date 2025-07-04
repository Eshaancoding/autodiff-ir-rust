use regex::Regex;

/// Number of spaces to replace each tab with
const TAB_WIDTH: usize = 4;

/// Replaces `\t` with fixed number of spaces
fn expand_tabs(s: &str) -> String {
    let mut result = String::new();
    let mut col = 0;
    for ch in s.chars() {
        if ch == '\t' {
            let spaces = TAB_WIDTH - (col % TAB_WIDTH);
            result.push_str(&" ".repeat(spaces));
            col += spaces;
        } else {
            result.push(ch);
            col += 1;
        }
    }
    result
}

/// Strips ANSI escape codes for alignment calculation
fn strip_ansi_codes(s: &str) -> String {
    let ansi_regex = Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    ansi_regex.replace_all(s, "").into_owned()
}

/// Displays two colored, possibly tabbed, multiline strings side-by-side
pub fn display_colored_side_by_side(left: String, right: String) {
    let left_lines: Vec<String> = left.lines().map(expand_tabs).collect();
    let right_lines: Vec<String> = right.lines().map(expand_tabs).collect();

    // Calculate visual width (without ANSI)
    let max_left_width = left_lines
        .iter()
        .map(|line| strip_ansi_codes(line).len())
        .max()
        .unwrap_or(0);

    let max_lines = left_lines.len().max(right_lines.len());

    for i in 0..max_lines {
        let left_line = left_lines.get(i).map(String::as_str).unwrap_or("");
        let right_line = right_lines.get(i).map(String::as_str).unwrap_or("");

        let stripped_len = strip_ansi_codes(left_line).len();
        let padding = max_left_width.saturating_sub(stripped_len);

        print!("{}", left_line);
        print!("{}", " ".repeat(padding));
        println!("       | {}", right_line);
    }
}

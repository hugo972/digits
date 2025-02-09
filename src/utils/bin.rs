use std::fs::File;
use std::io::Read;

const DIGIT_COUNT: usize = 1000;
const DIGIT_SIZE: usize = 28;
const DIGIT_BUFFER_SIZE: usize = DIGIT_SIZE.pow(2);

pub fn get_digits(path: &str) -> Vec<Vec<u8>> {
    let mut file = File::open(path).unwrap();
    let mut image_data = vec![0u8; DIGIT_COUNT * DIGIT_BUFFER_SIZE];
    file.read(&mut image_data).unwrap();
    image_data.chunks(DIGIT_BUFFER_SIZE).map(|d| d.to_vec()).collect()
}

pub fn get_bin_digits(path: &str) -> Vec<[i32; DIGIT_SIZE]> {
    let image_data = get_digits(path);
    let mut digits: Vec<[i32; DIGIT_SIZE]> = vec![[0; DIGIT_SIZE]; DIGIT_COUNT];
    for digit in 0..DIGIT_COUNT {
        for row in 0..DIGIT_SIZE {
            let mut d_row = 0;
            for col in 0..DIGIT_SIZE {
                if image_data[digit][row * DIGIT_SIZE + col] > 0 {
                    d_row |= 0x1;
                }

                d_row <<= 1;
            }

            digits[digit][row] = d_row;
        }
    }

    digits
}

pub fn print_bin_digit(digit: [i32; DIGIT_SIZE]) {
    for digit_row in digit.iter() {
        println!("{:#030b}", digit_row)
    }
}
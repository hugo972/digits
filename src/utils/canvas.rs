use pixel_canvas::{Canvas, Color};
use std::fs::File;
use std::io::Read;

pub fn render_digits(path: &str) {
    const DIGIT_COUNT: usize = 1000;
    const DIGIT_SIZE: usize = 28 * 28;

    let mut file = File::open(path).unwrap();
    let mut image_data = vec![0u8; DIGIT_COUNT * DIGIT_SIZE];
    file.read(&mut image_data).unwrap();

    let canvas = Canvas::new(1200, 1200).title("Digits");
    canvas.render(move |_mouse, image| {
        let width = image.width();
        let mut d_x = 0;
        let mut d_y = 0;
        for digit in 0..DIGIT_COUNT {
            for row in 0..28 {
                for col in 0..28 {
                    let color = image_data[digit * DIGIT_SIZE + row * 28 + col];
                    *image
                        .get_mut((1199 - (row + d_y)) * width + col + d_x)
                        .unwrap() = Color {
                        r: color,
                        g: color,
                        b: color,
                    }
                }
            }

            d_x += 30;
            if d_x % 1200 == 0 {
                d_y += 30;
                d_x = 0;
            }
        }
    });
}

use nannou::prelude::Rgb;
use nannou::prelude::*;
use std::fs::File;
use std::io::Read;

const DIGIT_COUNT: usize = 100;
const DIGIT_SIZE: usize = 28 * 28;

pub fn nan_main() {
    nannou::app(model).run();
}

struct Model {
    image_data: Vec<u8>,
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(512, 512)
        .title("Digits")
        .view(view)
        .build()
        .unwrap();

    let mut file = File::open("./data/data9.bin").unwrap();
    let mut image_data = vec![0u8; DIGIT_COUNT * DIGIT_SIZE];
    file.read(&mut image_data).unwrap();

    Model { image_data }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let win = app.window_rect();
    for i in 0..DIGIT_COUNT {
        const SIZE: f32 = 40.0;
        let row_digits = win.w() / SIZE;

        let r = Rect::from_x_y_w_h(
            (i as f32 % row_digits) * SIZE - win.w() / 2.0,
            (i as f32 / row_digits) * SIZE,
            SIZE,
            SIZE,
        );

        draw_digit(&draw, &model, i, r);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_digit(draw: &Draw, model: &Model, digit: usize, rect: Rect) {
    let (x, y) = rect.x_y();
    let p_w = rect.w() / 28.0;
    let p_h = rect.h() / 28.0;
    for row in 0..28 {
        for col in 0..28 {
            let color = model.image_data[digit * DIGIT_SIZE + col * 28 + row];
            draw.rect()
                .x_y(x + p_w * row as f32, y + p_h * col as f32)
                .w_h(p_w, p_h)
                .color(Rgb::new(color, color, color));
        }
    }
}

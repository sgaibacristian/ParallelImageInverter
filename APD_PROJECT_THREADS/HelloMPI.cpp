#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

using namespace std;

void invertColors(cv::Mat& image, int startRow, int endRow) {
    // Obținerea dimensiunilor imaginii
    int cols = image.cols;

    // Parcurgerea regiunii de pixeli și inversarea culorilor
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            pixel[0] = 255 - pixel[0]; // canalul albastru
            pixel[1] = 255 - pixel[1]; // canalul verde
            pixel[2] = 255 - pixel[2]; // canalul roșu
        }
    }
}

void processChunk(cv::Mat& image, int startRow, int endRow, int thread_id) {
    // Inversarea culorilor în regiunea specificată
    invertColors(image, startRow, endRow);

    // Salvarea bucății procesate într-un fișier separat
    cv::Mat chunk = image(cv::Range(startRow, endRow), cv::Range::all());
    string filename = "image_chunk_" + to_string(thread_id) + ".jpg";
    cv::imwrite(filename, chunk);
    cout << "Thread " << thread_id << " saved chunk to " << filename << endl;
}

int main(int argc, char** argv) {
    const auto start = chrono::high_resolution_clock::now();

    // Citirea imaginii
    cv::Mat image = cv::imread("input.jpg");

    // Verificarea dacă imaginea a fost citită corect
    if (image.empty()) {
        cerr << "Could not read the image." << endl;
        return -1;
    }

    // Obținerea dimensiunilor imaginii
    int rows = image.rows;
    int cols = image.cols;

    // Numărul de thread-uri utilizate
    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;

    // Calcularea numărului de rânduri pe care fiecare thread le va trata
    int rows_per_thread = rows / num_threads;
    int remainder = rows % num_threads; // Rândurile rămase

    // Crearea și pornirea thread-urilor
    for (int i = 0; i < num_threads; ++i) {
        int startRow = i * rows_per_thread + min(i, remainder);
        int endRow = startRow + rows_per_thread + (i < remainder ? 1 : 0);
        threads.push_back(thread(processChunk, ref(image), startRow, endRow, i));
    }

    // Așteptarea finalizării thread-urilor
    for (auto& t : threads) {
        t.join();
    }

    // Salvarea imaginii inverse complete
    cv::imwrite("inverted_image.jpg", image);
    const auto end = chrono::high_resolution_clock::now();
    const chrono::duration<double> elapsed_time = end - start;
    cout << "Execution time: " << elapsed_time.count() << " seconds." << endl;

    // Afișarea imaginii inverse într-o fereastră
    cv::imshow("Inverted Image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

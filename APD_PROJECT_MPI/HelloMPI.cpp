#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
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

int main(int argc, char** argv) {
    // Inițializarea mediului MPI
    MPI_Init(&argc, &argv);
    const clock_t start = clock();
    // Obținerea numărului total de procese și a ID-ului procesului curent
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cout << "Process " << world_rank << " initialized." << endl;

    // Citirea imaginii doar de către procesul cu ID-ul 0
    cv::Mat image;
    if (world_rank == 0) {
        cout << "Process 0 reading the image..." << endl;
        image = cv::imread("input.jpg");

        // Verificarea dacă imaginea a fost citită corect
        if (image.empty()) {
            cerr << "Could not read the image." << endl;
            MPI_Finalize();
            return -1;
        }
        cout << "Process 0 read the image." << endl;
    }

    // Transmiterea dimensiunilor imaginii de la procesul cu ID-ul 0 către toate celelalte procese
    int rows = 0, cols = 0;
    if (world_rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Process " << world_rank << " received image dimensions." << endl;

    // Calcularea numărului de rânduri pe care fiecare proces le va trata
    int rows_per_process = rows / world_size;
    int remainder = rows % world_size; // Rândurile rămase

    // Determinarea benzii de rânduri atribuite fiecărui proces
    int startRow = world_rank * rows_per_process + min(world_rank, remainder);
    int endRow = startRow + rows_per_process + (world_rank < remainder ? 1 : 0);

    cout << "Process " << world_rank << " will process rows from " << startRow << " to " << endRow - 1 << endl;

    // Citirea și inversarea culorilor pixelilor în paralel
    if (world_rank == 0) {
        // Transmiterea regiunii de pixeli proceselor non-zero
        for (int i = 1; i < world_size; ++i) {
            int process_start_row = i * rows_per_process + min(i, remainder);
            int process_end_row = process_start_row + rows_per_process + (i < remainder ? 1 : 0);
            int region_rows = process_end_row - process_start_row;
            MPI_Send(image.ptr(process_start_row), region_rows * cols * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
        }

        cout << "Process 0 sent image data to other processes." << endl;

        // Inversarea culorilor în regiunea proprie
        invertColors(image, startRow, endRow);

        cout << "Process 0 processed its own region." << endl;

        // Primirea regiunilor inverse de la celelalte procese
        for (int i = 1; i < world_size; ++i) {
            int process_start_row = i * rows_per_process + min(i, remainder);
            int process_end_row = process_start_row + rows_per_process + (i < remainder ? 1 : 0);
            int region_rows = process_end_row - process_start_row;
            MPI_Recv(image.ptr(process_start_row), region_rows * cols * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        cout << "Process 0 received processed regions from other processes." << endl;
    }
    else {
        // Primirea regiunii de pixeli de la procesul cu ID-ul 0
        int region_rows = endRow - startRow;
        cv::Mat image_chunk(region_rows, cols, CV_8UC3);
        MPI_Recv(image_chunk.data, region_rows * cols * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Process " << world_rank << " received image data." << endl;

        // Inversarea culorilor în regiunea primită
        invertColors(image_chunk, 0, region_rows);
		cv::imwrite(("image_chunk_" + to_string(world_rank) + ".jpg") ,image_chunk);
        cout << "Process " << world_rank << " processed its region." << endl;

        // Trimiterea înapoi a regiunii inverse la procesul cu ID-ul 0
        MPI_Send(image_chunk.data, region_rows * cols * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);

        cout << "Process " << world_rank << " sent processed region back to process 0." << endl;
    }

    // Finalizarea mediului MPI
    MPI_Finalize();

    // Salvarea imaginii inverse doar de către procesul cu ID-ul 0
    if (world_rank == 0) {
        cv::imwrite("inverted_image.jpg", image);
        const clock_t end = clock();
        const double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        cout << "Execution time: "<< elapsed_time;
        cout << "Process 0 saved the inverted image." << endl;

        // Afișarea imaginii inverse într-o fereastră
        cv::imshow("Inverted Image", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}

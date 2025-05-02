import { Component, OnInit } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { ChartConfiguration, ChartOptions } from "chart.js";

@Component({
  selector: "app-dashboard",
  template: `
    <div class="dashboard-container">
      <mat-card class="prediction-card">
        <mat-card-header>
          <mat-card-title>Real-time RUL Prediction</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="upload-section">
            <input
              type="file"
              (change)="onFileSelected($event)"
              accept=".csv,.txt"
            />
            <button mat-raised-button color="primary" (click)="uploadFile()">
              Upload Data
            </button>
          </div>

          <div class="prediction-section" *ngIf="predictionData">
            <h3>Engine Unit: {{ predictionData.unitId }}</h3>
            <h2>Predicted RUL: {{ predictionData.rul }} cycles</h2>
            <div class="chart-container">
              <canvas
                baseChart
                [data]="lineChartData"
                [options]="lineChartOptions"
                [type]="'line'"
              >
              </canvas>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <mat-card class="confidence-card">
        <mat-card-header>
          <mat-card-title>Confidence Intervals</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="confidence-section" *ngIf="confidenceData">
            <div class="confidence-bar">
              <div
                class="confidence-fill"
                [style.width.%]="confidenceData.confidence * 100"
              >
                {{ confidenceData.confidence * 100 | number : "1.0-0" }}%
              </div>
            </div>
            <p>
              Model Confidence:
              {{ confidenceData.confidence * 100 | number : "1.0-0" }}%
            </p>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [
    `
      .dashboard-container {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
        padding: 20px;
      }

      .upload-section {
        margin: 20px 0;
        display: flex;
        gap: 10px;
        align-items: center;
      }

      .prediction-section {
        margin-top: 20px;
      }

      .chart-container {
        height: 300px;
        margin-top: 20px;
      }

      .confidence-section {
        margin-top: 20px;
      }

      .confidence-bar {
        width: 100%;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
      }

      .confidence-fill {
        height: 100%;
        background-color: #4caf50;
        color: white;
        text-align: center;
        line-height: 20px;
        transition: width 0.3s ease;
      }
    `,
  ],
})
export class DashboardComponent implements OnInit {
  selectedFile: File | null = null;
  predictionData: any = null;
  confidenceData: any = null;

  lineChartData: ChartConfiguration<"line">["data"] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: "RUL Prediction",
        fill: true,
        tension: 0.5,
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
    ],
  };

  lineChartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "RUL (cycles)",
        },
      },
      x: {
        title: {
          display: true,
          text: "Time",
        },
      },
    },
  };

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    // Initialize any required data
  }

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
  }

  uploadFile(): void {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append("file", this.selectedFile);

    this.http.post("/api/predict", formData).subscribe({
      next: (response: any) => {
        this.predictionData = response.prediction;
        this.confidenceData = response.confidence;

        // Update chart data
        this.lineChartData.labels = response.timestamps;
        this.lineChartData.datasets[0].data = response.predictions;
      },
      error: (error) => {
        console.error("Error uploading file:", error);
      },
    });
  }
}

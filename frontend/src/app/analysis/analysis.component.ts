import { Component, OnInit } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { ChartConfiguration, ChartOptions } from "chart.js";

@Component({
  selector: "app-analysis",
  template: `
    <div class="analysis-container">
      <mat-card class="feature-importance">
        <mat-card-header>
          <mat-card-title>Feature Importance Analysis</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="chart-container">
            <canvas
              baseChart
              [data]="featureImportanceData"
              [options]="featureImportanceOptions"
              [type]="'bar'"
            >
            </canvas>
          </div>
        </mat-card-content>
      </mat-card>

      <mat-card class="correlation-matrix">
        <mat-card-header>
          <mat-card-title>Sensor Correlation Matrix</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="chart-container">
            <canvas
              baseChart
              [data]="correlationData"
              [options]="correlationOptions"
              [type]="'heatmap'"
            >
            </canvas>
          </div>
        </mat-card-content>
      </mat-card>

      <mat-card class="explanation">
        <mat-card-header>
          <mat-card-title>Model Explanation</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="explanation-content" *ngIf="explanationData">
            <div class="sensor-contributions">
              <h3>Top Contributing Sensors</h3>
              <mat-list>
                <mat-list-item
                  *ngFor="let sensor of explanationData.top_sensors"
                >
                  <span matListItemTitle>{{ sensor.name }}</span>
                  <span matListItemLine
                    >Contribution:
                    {{ sensor.contribution | number : "1.2-2" }}</span
                  >
                </mat-list-item>
              </mat-list>
            </div>

            <div class="degradation-patterns">
              <h3>Degradation Patterns</h3>
              <div class="pattern-chart">
                <canvas
                  baseChart
                  [data]="degradationPatternData"
                  [options]="degradationPatternOptions"
                  [type]="'line'"
                >
                </canvas>
              </div>
            </div>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [
    `
      .analysis-container {
        padding: 20px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
      }

      .chart-container {
        height: 300px;
        margin: 20px 0;
      }

      .explanation-content {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 20px;
      }

      .pattern-chart {
        height: 200px;
        margin-top: 20px;
      }

      mat-card {
        margin-bottom: 20px;
      }
    `,
  ],
})
export class AnalysisComponent implements OnInit {
  featureImportanceData: ChartConfiguration<"bar">["data"] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: "Feature Importance",
        backgroundColor: "#4CAF50",
      },
    ],
  };

  featureImportanceOptions: ChartOptions<"bar"> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: "y",
    scales: {
      x: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Importance Score",
        },
      },
    },
  };

  correlationData: ChartConfiguration<"heatmap">["data"] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: "Correlation Matrix",
      },
    ],
  };

  correlationOptions: ChartOptions<"heatmap"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
  };

  degradationPatternData: ChartConfiguration<"line">["data"] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: "Degradation Pattern",
        borderColor: "#FF6384",
        fill: false,
      },
    ],
  };

  degradationPatternOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Sensor Value",
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

  explanationData: any = null;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadFeatureImportance();
    this.loadCorrelationMatrix();
    this.loadModelExplanation();
  }

  loadFeatureImportance(): void {
    this.http.get("/api/explain/features").subscribe({
      next: (response: any) => {
        this.featureImportanceData.labels = response.features;
        this.featureImportanceData.datasets[0].data =
          response.importance_scores;
      },
      error: (error) => {
        console.error("Error loading feature importance:", error);
      },
    });
  }

  loadCorrelationMatrix(): void {
    this.http.get("/api/explain/correlation").subscribe({
      next: (response: any) => {
        this.correlationData.labels = response.sensors;
        this.correlationData.datasets[0].data = response.correlation_matrix;
      },
      error: (error) => {
        console.error("Error loading correlation matrix:", error);
      },
    });
  }

  loadModelExplanation(): void {
    this.http.get("/api/explain/model").subscribe({
      next: (response: any) => {
        this.explanationData = response;
        this.updateDegradationPattern(response.degradation_pattern);
      },
      error: (error) => {
        console.error("Error loading model explanation:", error);
      },
    });
  }

  updateDegradationPattern(pattern: any): void {
    this.degradationPatternData.labels = pattern.timestamps;
    this.degradationPatternData.datasets[0].data = pattern.values;
  }
}

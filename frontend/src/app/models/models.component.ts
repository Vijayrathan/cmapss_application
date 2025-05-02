import { Component, OnInit } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { MatTableDataSource } from "@angular/material/table";
import { ChartConfiguration, ChartOptions } from "chart.js";

interface ModelInfo {
  name: string;
  type: string;
  performance: {
    rmse: number;
    mae: number;
    r2: number;
  };
  lastUpdated: string;
  status: string;
}

@Component({
  selector: "app-models",
  template: `
    <div class="models-container">
      <mat-card>
        <mat-card-header>
          <mat-card-title>Available Models</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <mat-table [dataSource]="dataSource">
            <ng-container matColumnDef="name">
              <mat-header-cell *matHeaderCellDef>Model Name</mat-header-cell>
              <mat-cell *matCellDef="let model">{{ model.name }}</mat-cell>
            </ng-container>

            <ng-container matColumnDef="type">
              <mat-header-cell *matHeaderCellDef>Type</mat-header-cell>
              <mat-cell *matCellDef="let model">{{ model.type }}</mat-cell>
            </ng-container>

            <ng-container matColumnDef="performance">
              <mat-header-cell *matHeaderCellDef>Performance</mat-header-cell>
              <mat-cell *matCellDef="let model">
                RMSE: {{ model.performance.rmse | number : "1.2-2" }} | MAE:
                {{ model.performance.mae | number : "1.2-2" }} | R²:
                {{ model.performance.r2 | number : "1.2-2" }}
              </mat-cell>
            </ng-container>

            <ng-container matColumnDef="lastUpdated">
              <mat-header-cell *matHeaderCellDef>Last Updated</mat-header-cell>
              <mat-cell *matCellDef="let model">{{
                model.lastUpdated
              }}</mat-cell>
            </ng-container>

            <ng-container matColumnDef="status">
              <mat-header-cell *matHeaderCellDef>Status</mat-header-cell>
              <mat-cell *matCellDef="let model">
                <mat-chip
                  [color]="model.status === 'active' ? 'primary' : 'warn'"
                >
                  {{ model.status }}
                </mat-chip>
              </mat-cell>
            </ng-container>

            <ng-container matColumnDef="actions">
              <mat-header-cell *matHeaderCellDef>Actions</mat-header-cell>
              <mat-cell *matCellDef="let model">
                <button mat-icon-button (click)="selectModel(model)">
                  <mat-icon>play_arrow</mat-icon>
                </button>
                <button mat-icon-button (click)="retrainModel(model)">
                  <mat-icon>refresh</mat-icon>
                </button>
              </mat-cell>
            </ng-container>

            <mat-header-row
              *matHeaderRowDef="displayedColumns"
            ></mat-header-row>
            <mat-row *matRowDef="let row; columns: displayedColumns"></mat-row>
          </mat-table>
        </mat-card-content>
      </mat-card>

      <div class="model-details" *ngIf="selectedModel">
        <mat-card>
          <mat-card-header>
            <mat-card-title>{{ selectedModel.name }} Details</mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div class="performance-chart">
              <canvas
                baseChart
                [data]="performanceChartData"
                [options]="performanceChartOptions"
                [type]="'bar'"
              >
              </canvas>
            </div>

            <div class="drift-monitor" *ngIf="driftData">
              <h3>Model Drift Monitor</h3>
              <div
                class="drift-indicator"
                [class.drift-detected]="driftData.drift_detected"
              >
                <p>
                  Drift Status:
                  {{ driftData.drift_detected ? "Detected" : "Normal" }}
                </p>
                <p>
                  Confidence:
                  {{ driftData.confidence * 100 | number : "1.0-0" }}%
                </p>
              </div>
            </div>
          </mat-card-content>
        </mat-card>
      </div>
    </div>
  `,
  styles: [
    `
      .models-container {
        padding: 20px;
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
      }

      .performance-chart {
        height: 300px;
        margin: 20px 0;
      }

      .drift-monitor {
        margin-top: 20px;
        padding: 10px;
        border-radius: 4px;
      }

      .drift-indicator {
        padding: 10px;
        background-color: #4caf50;
        color: white;
        border-radius: 4px;
      }

      .drift-indicator.drift-detected {
        background-color: #f44336;
      }

      mat-table {
        width: 100%;
      }
    `,
  ],
})
export class ModelsComponent implements OnInit {
  displayedColumns: string[] = [
    "name",
    "type",
    "performance",
    "lastUpdated",
    "status",
    "actions",
  ];
  dataSource = new MatTableDataSource<ModelInfo>();
  selectedModel: ModelInfo | null = null;
  driftData: any = null;

  performanceChartData: ChartConfiguration<"bar">["data"] = {
    labels: ["RMSE", "MAE", "R²"],
    datasets: [
      {
        data: [],
        label: "Performance Metrics",
        backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56"],
      },
    ],
  };

  performanceChartOptions: ChartOptions<"bar"> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadModels();
  }

  loadModels(): void {
    this.http.get("/api/models").subscribe({
      next: (models: any) => {
        this.dataSource.data = models;
      },
      error: (error) => {
        console.error("Error loading models:", error);
      },
    });
  }

  selectModel(model: ModelInfo): void {
    this.selectedModel = model;
    this.updatePerformanceChart(model);
    this.checkDrift(model);
  }

  updatePerformanceChart(model: ModelInfo): void {
    this.performanceChartData.datasets[0].data = [
      model.performance.rmse,
      model.performance.mae,
      model.performance.r2,
    ];
  }

  checkDrift(model: ModelInfo): void {
    this.http.get(`/api/drift/${model.name}`).subscribe({
      next: (drift: any) => {
        this.driftData = drift;
      },
      error: (error) => {
        console.error("Error checking drift:", error);
      },
    });
  }

  retrainModel(model: ModelInfo): void {
    this.http.post(`/api/retrain/${model.name}`, {}).subscribe({
      next: (response: any) => {
        console.log("Retraining initiated:", response);
        // Refresh model list after retraining
        setTimeout(() => this.loadModels(), 5000);
      },
      error: (error) => {
        console.error("Error initiating retraining:", error);
      },
    });
  }
}

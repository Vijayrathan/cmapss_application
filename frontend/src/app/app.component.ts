import { Component } from "@angular/core";
import { RouterOutlet } from "@angular/router";

@Component({
  selector: "app-root",
  template: `
    <mat-toolbar color="primary">
      <span>Turbofan RUL Prediction System</span>
      <span class="spacer"></span>
      <button mat-button routerLink="/dashboard">Dashboard</button>
      <button mat-button routerLink="/models">Models</button>
      <button mat-button routerLink="/analysis">Analysis</button>
    </mat-toolbar>

    <div class="content">
      <router-outlet></router-outlet>
    </div>
  `,
  styles: [
    `
      .spacer {
        flex: 1 1 auto;
      }
      .content {
        padding: 20px;
      }
    `,
  ],
})
export class AppComponent {
  title = "turbofan-rul-frontend";
}

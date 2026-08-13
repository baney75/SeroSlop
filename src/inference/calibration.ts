export interface PlattCalibration {
  slope: number;
  intercept: number;
}

function clampProbability(value: number): number {
  return Math.min(1 - 1e-7, Math.max(1e-7, value));
}

export function calibrateAiLikelihood(rawAiLikelihood: number, calibration: PlattCalibration): number {
  const probability = clampProbability(rawAiLikelihood);
  const rawLogit = Math.log(probability / (1 - probability));
  const calibratedLogit = Math.max(-40, Math.min(40, calibration.slope * rawLogit + calibration.intercept));
  return 1 / (1 + Math.exp(-calibratedLogit));
}

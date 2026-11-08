import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 120000,
  headers: {
    "Content-Type": "application/json",
  },
});

export const checkBackend = async () => {
  const response = await api.get("/health");
  return response.data;
};

export const getModelInfo = async () => {
  const response = await api.get("/model-info");
  return response.data;
};

export const trainModel = async (payload) => {
  const response = await api.post("/fit", payload);
  return response.data;
};

export const predictITE = async (payload) => {
  const response = await api.post("/predict", payload);
  return response.data;
};
export const evaluateUplift = async (payload) => {
  const response = await api.post("/uplift", payload);
  return response.data;
};

export const optimizeBudget = async (payload) => {
  const response = await api.post("/optimize", payload);
  return response.data;
};

export const runPSM = async (payload) => {
  const response = await api.post("/psm", payload);
  return response.data;
};

export const runDoWhy = async (payload) => {
  const response = await api.post("/dowhy", payload);
  return response.data;
};

export const runDrift = async (payload) => {
  const response = await api.post("/drift", payload);
  return response.data;
};

export const generateStrategySummary = async (payload) => {
  const response = await api.post("/strategy-summary", payload);
  return response.data;
};

export default api;

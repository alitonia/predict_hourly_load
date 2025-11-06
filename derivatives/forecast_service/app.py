import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging

from google import genai

from starlette.requests import Request
from starlette.responses import JSONResponse
import hashlib

# Load environment variables
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
lgb_model = None
prophet_model = None
prophet_simple_model = None
model_metadata = None
gemini_model = None

# Models directory
relative_model_dir = "../../models"
MODELS_DIR = Path(os.path.join(os.getcwd(), relative_model_dir))

# Public directory for static files
PUBLIC_DIR = Path(os.path.join(os.getcwd(), "public"))
# Create a public directory if it doesn't exist
PUBLIC_DIR.mkdir(exist_ok=True)

temp_report_mapper = dict()


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global lgb_model, prophet_model, prophet_simple_model, model_metadata, gemini_model

    try:
        logger.info("=" * 80)
        logger.info("LOADING MODELS...")
        logger.info("=" * 80)

        # Load LightGBM model
        lgb_path = MODELS_DIR / "lightgbm_power_forecast.pkl"
        if lgb_path.exists():
            lgb_model = joblib.load(lgb_path)
            logger.info(f"✓ LightGBM model loaded from {lgb_path}")
        else:
            logger.warning(f"⚠ LightGBM model not found at {lgb_path}")

        # Load Prophet model with holidays
        prophet_path = MODELS_DIR / "prophet_power_forecast_with_holidays.pkl"
        if prophet_path.exists():
            prophet_model = joblib.load(prophet_path)
            logger.info(f"✓ Prophet model (with holidays) loaded from {prophet_path}")
        else:
            logger.warning(f"⚠ Prophet model not found at {prophet_path}")

        # Load simple Prophet model
        prophet_simple_path = MODELS_DIR / "prophet_power_forecast_simple.pkl"
        if prophet_simple_path.exists():
            prophet_simple_model = joblib.load(prophet_simple_path)
            logger.info(f"✓ Prophet model (simple) loaded from {prophet_simple_path}")
        else:
            logger.warning(f"⚠ Simple Prophet model not found at {prophet_simple_path}")

        # Load metadata
        metadata_path = MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"✓ Model metadata loaded from {metadata_path}")
        else:
            logger.warning(f"⚠ Model metadata not found at {metadata_path}")
            model_metadata = {}

        logger.info("=" * 80)
        logger.info("API SERVER READY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("=" * 80)
    logger.info("SHUTTING DOWN API SERVER")
    logger.info("=" * 80)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Power Demand Forecasting API",
    description="API for predicting power demand using LightGBM and Prophet models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD HH:MM:SS format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD HH:MM:SS format")
    model_type: Literal["lightgbm", "prophet", "prophet_simple", "ensemble"] = Field(
        default="lightgbm",
        description="Model to use for prediction"
    )

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD HH:MM:SS format')

    @field_validator('end_date')
    @classmethod
    def validate_end_after_start(cls, v: str, info) -> str:
        if 'start_date' in info.data:
            start = datetime.strptime(info.data['start_date'], '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            if end <= start:
                raise ValueError('end_date must be after start_date')
        return v


class ReportRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD HH:MM:SS format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD HH:MM:SS format")
    model_type: Literal["lightgbm", "prophet", "prophet_simple", "ensemble"] = Field(
        default="lightgbm",
        description="Model to use for prediction"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context or requirements for the report"
    )

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD HH:MM:SS format')


class SinglePrediction(BaseModel):
    timestamp: str
    predicted_power_mw: float
    model_used: str


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    summary: dict
    metadata: dict


class ReportResponse(BaseModel):
    report: str
    prediction_summary: dict
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    api_version: str
    gemini_available: bool


class ModelInfoResponse(BaseModel):
    lightgbm: dict
    prophet_with_holidays: dict
    prophet_simple: dict
    comparison: dict


# Helper function to create features for LightGBM
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour.astype('int32')
    df['dayofweek'] = df.index.dayofweek.astype('int32')
    df['quarter'] = df.index.quarter.astype('int32')
    df['month'] = df.index.month.astype('int32')
    df['year'] = df.index.year.astype('int32')
    df['dayofyear'] = df.index.dayofyear.astype('int32')
    df['dayofmonth'] = df.index.day.astype('int32')
    df['weekofyear'] = df.index.isocalendar().week.astype('int32')

    # Add season
    df['season'] = df.index.month.map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,  # Spring
        6: 2, 7: 2, 8: 2,  # Summer
        9: 3, 10: 3, 11: 3  # Fall
    }).astype('int32')

    return df


# API Endpoints
@app.get("/", response_model=dict)
async def root(request: Request):
    """Serve the main HTML page or return API info based on Accept header"""
    # Check Accept header
    accept_header = request.headers.get("accept", "")

    # If a client explicitly requests JSON, return API info
    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(content={
            "message": "Power Demand Forecasting API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "make_report": "/make-report",
                "model_info": "/model-info",
                "available_models": "/available-models",
                "docs": "/docs"
            }
        })

    # Otherwise, try to serve index.html
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback to API information if no index.html exists
        return JSONResponse(content={
            "message": "Power Demand Forecasting API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "make_report": "/make-report",
                "model_info": "/model-info",
                "available_models": "/available-models",
                "docs": "/docs"
            }
        })


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "lightgbm": lgb_model is not None,
            "prophet_with_holidays": prophet_model is not None,
            "prophet_simple": prophet_simple_model is not None
        },
        api_version="1.0.0",
        gemini_available=gemini_model is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models"""
    if not model_metadata:
        raise HTTPException(status_code=500, detail="Model metadata not available")

    return ModelInfoResponse(**model_metadata)


@app.get("/available-models")
async def get_available_models():
    """Get a list of available models"""
    return {"models": list(model_metadata.keys())}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make power demand predictions for a given time range

    - **start_date**: Start date for predictions (YYYY-MM-DD HH:MM:SS)
    - **end_date**: End date for predictions (YYYY-MM-DD HH:MM:SS)
    - **model_type**: Model to use (lightgbm, prophet, prophet_simple, ensemble)
    """
    try:
        # Parse dates
        start_dt = pd.to_datetime(request.start_date)
        end_dt = pd.to_datetime(request.end_date)

        # Create datetime range (hourly)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='h')

        if len(date_range) == 0:
            raise HTTPException(status_code=400, detail="Invalid date range")

        if len(date_range) > 8760:  # More than 1 year
            raise HTTPException(
                status_code=400,
                detail="Date range too large. Maximum 1 year (8760 hours) allowed"
            )

        # Make predictions based on the model type
        if request.model_type == "lightgbm":
            if lgb_model is None:
                raise HTTPException(status_code=500, detail="LightGBM model not loaded")
            predictions = predict_lightgbm(date_range)
            model_used = "LightGBM"

        elif request.model_type == "prophet":
            if prophet_model is None:
                raise HTTPException(status_code=500, detail="Prophet model not loaded")
            predictions = predict_prophet(date_range, prophet_model)
            model_used = "Prophet (with holidays)"

        elif request.model_type == "prophet_simple":
            if prophet_simple_model is None:
                raise HTTPException(status_code=500, detail="Simple Prophet model not loaded")
            predictions = predict_prophet(date_range, prophet_simple_model)
            model_used = "Prophet (simple)"

        elif request.model_type == "ensemble":
            if lgb_model is None or prophet_model is None:
                raise HTTPException(status_code=500, detail="Ensemble requires both models")
            lgb_pred = predict_lightgbm(date_range)
            prophet_pred = predict_prophet(date_range, prophet_model)
            # Simple average ensemble
            predictions = (lgb_pred + prophet_pred) / 2
            model_used = "Ensemble (LightGBM + Prophet)"

        # Create response
        prediction_list = [
            SinglePrediction(
                timestamp=str(ts),
                predicted_power_mw=float(pred),
                model_used=model_used
            )
            for ts, pred in zip(date_range, predictions)
        ]

        # Calculate summary statistics
        summary = {
            "total_predictions": len(predictions),
            "start_date": str(date_range[0]),
            "end_date": str(date_range[-1]),
            "mean_power_mw": float(np.mean(predictions)),
            "min_power_mw": float(np.min(predictions)),
            "max_power_mw": float(np.max(predictions)),
            "std_power_mw": float(np.std(predictions)),
            "peak_hour": str(date_range[np.argmax(predictions)]),
            "lowest_hour": str(date_range[np.argmin(predictions)])
        }

        # Metadata
        metadata = {
            "model_type": request.model_type,
            "model_used": model_used,
            "prediction_timestamp": str(datetime.now()),
            "frequency": "hourly"
        }

        return PredictionResponse(
            predictions=prediction_list,
            summary=summary,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/make-report", response_model=ReportResponse)
async def make_report(request: ReportRequest):
    """
    Generate a capacity planning report using Gemini AI
    
    - **start_date**: Start date for predictions (YYYY-MM-DD HH:MM:SS)
    - **end_date**: End date for predictions (YYYY-MM-DD HH:MM:SS)
    - **model_type**: Model to use for predictions
    - **additional_context**: Additional context for the report
    """
    try:

        # Check in cache
        cache_key = f"{request.start_date}_{request.end_date}_{request.model_type}"
        if cache_key in temp_report_mapper:
            return temp_report_mapper[cache_key]

        # First get predictions
        prediction_request = PredictionRequest(
            start_date=request.start_date,
            end_date=request.end_date,
            model_type=request.model_type
        )
        signature = hashlib.md5(f"{request.start_date}-{request.end_date}-{request.model_type}".encode())
        cache_key = f"report-{signature.hexdigest()}"

        print("Making predictions...")
        prediction_response = await predict(prediction_request)

        # Prepare data for a report
        summary = prediction_response.summary

        # Create a prompt for Gemini
        prompt = f"""
        Generate a comprehensive capacity planning brief based on the following power demand forecast data:
        
        Time Period: {summary['start_date']} to {summary['end_date']}
        Total Predictions: {summary['total_predictions']}
        Average Power Demand: {summary['mean_power_mw']:.2f} MW
        Peak Power Demand: {summary['max_power_mw']:.2f} MW at {summary['peak_hour']}
        Lowest Power Demand: {summary['min_power_mw']:.2f} MW at {summary['lowest_hour']}
        Standard Deviation: {summary['std_power_mw']:.2f} MW
        
        Model Used: {prediction_response.metadata['model_used']}
        
        Additional Context: {request.additional_context or 'No additional context provided'}
        
        Please provide:
        1. Executive Summary
        2. Key Findings
        3. Capacity Planning Recommendations
        4. Risk Assessment
        5. Action Items
        
        Format your response in clear sections with appropriate headings.
        Return the response in MARKDOWN format.
        """

        print("Generating report...")
        # Generate report using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="Failed to generate report from Gemini")

        return_obj = ReportResponse(
            report=response.text,
            prediction_summary=summary,
            metadata={
                "generated_at": str(datetime.now()),
                "model_used": prediction_response.metadata['model_used'],
                "prompt_used": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "signature": cache_key
            }
        )
        temp_report_mapper[cache_key] = return_obj

        return return_obj


    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# Add this new endpoint to retrieve cached reports
@app.get("/get-report/{signature}")
async def get_report(signature: str):
    """
    Retrieve a cached report by its signature
    
    - **signature**: The unique signature of the report
    """
    try:
        if signature not in temp_report_mapper:
            raise HTTPException(status_code=404, detail="Report not found or expired")

        return temp_report_mapper[signature]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def predict_lightgbm(date_range: pd.DatetimeIndex) -> np.ndarray:
    """Make predictions using the LightGBM model"""
    # Create DataFrame with datetime index
    df = pd.DataFrame(index=date_range)

    # Create features
    df = create_features(df)

    # Get feature columns from metadata
    if model_metadata and 'lightgbm' in model_metadata:
        feature_cols = model_metadata['lightgbm']['features']
    else:
        print(1)
        # Default features if metadata not available
        feature_cols = ['hour', 'dayofweek', 'quarter', 'month', 'year',
                        'dayofyear', 'dayofmonth', 'weekofyear', 'season']

    # Ensure all required features are present and in the correct order
    X = df[feature_cols]
    # convert season to categorical
    X['season'] = X['month'].apply(get_season).astype('category')

    # Make predictions
    predictions = lgb_model.predict(
        X
        , num_iteration=lgb_model.best_iteration_
    )

    return predictions


def predict_prophet(date_range: pd.DatetimeIndex, model) -> np.ndarray:
    """Make predictions using Prophet model"""
    # Create a future dataframe for Prophet
    future = pd.DataFrame({'ds': date_range})

    # Make predictions
    forecast = model.predict(future)

    # Return predictions (yhat column)
    return forecast['yhat'].values


# Create get api to list current key in cache
@app.get("/cache_keys")
async def list_cache_keys():
    return list(temp_report_mapper.keys())


# Mount static files
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

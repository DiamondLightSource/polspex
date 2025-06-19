import os 
import logging 
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import msgpack
import numpy as np

import secrets
from subprocess import Popen
# from contextlib import asynccontextmanager
# import nest_asyncio
# import requests

from .environment import AVAILABLE_EXPIDS, get_path_filespec, get_beamline, get_quanty_path
from .parameters import AVAILABLE_SYMMETRIES, AVAILABLE_DQ
from .xas_analysis import find_pairs, gen_metadata_str, find_similar_measurements
from .plot_models import lineProps
from .quanty_runner import gen_simulation


# Generate a secure token
jupyter_token = secrets.token_hex(32)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SimulationInputs(BaseModel):
    ion: str
    charge: str
    symmetry: str
    beta: float
    tenDq: dict[str, float]  # Dq values for each symmetry
    bFieldX: float
    bFieldY: float
    bFieldZ: float
    hFieldX: float
    hFieldY: float
    hFieldZ: float
    temperature: float
    path: str


class Simulations(BaseModel):
    sims: list[SimulationInputs]


class SimulationOutputs(BaseModel):
    message: str
    table: str
    plot1: lineProps
    plot2: lineProps


class AvailableCharges(BaseModel):
    charge: list[str]  # symmetries

class AvailableElements(BaseModel):
    ion: AvailableCharges

class DqConfiguration(BaseModel):
    conf: float

class DqParameters(BaseModel):
    initial: DqConfiguration  # keys are Dq values (e.g., "10Dq", "Dmu")
    final: DqConfiguration

class SymmetryDq(BaseModel):
    symmetry: DqParameters  # Keys are symmetries (e.g., "Oh", "Td")

class ChargeDq(BaseModel):
    charge: SymmetryDq  # Keys are charge states (e.g., "2+", "3+")

class AvailableDq(BaseModel):
    """
    Description of the Dq values for each element.
    """
    ion: ChargeDq

class DataPath(BaseModel):
    path: str

class DataFiles(BaseModel):
    files: list[str]

class LoadMeasuredData(BaseModel):
    files: list[str]
    background_type: str

class MeasuredData(BaseModel):
    pol_pairs: list[lineProps]
    average: lineProps 
    table: str
    element: str
    field: list[float, float, float]
    temperature: float

class LoadMetadata(BaseModel):
    files: dict[int, str]


def encoder(obj) -> dict[str, Any]:
    if isinstance(obj, np.ndarray):
        logger.info(f"Encoding numpy array: {obj.dtype} {obj.dtype.kind} {obj.size} {obj.shape}")
        # Create javascript NDarray like object
        obj = dict(
            nd=True, dtype=obj.dtype.str, shape=obj.shape, data=obj.data.tolist()
        )
        # logger.info(f"Encoded numpy array: {obj}")
    return obj


########################################################
#################### FastAPI App #######################
########################################################


def create_fastapi_app():
    """
    Create a FastAPI application instance.
    """
    app = FastAPI(title="PolSpeX FastAPI", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #@app.get("/api/elements", response_model=AvailableElements)
    @app.get("/api/elements")
    async def get_element():
        return AVAILABLE_SYMMETRIES

    # @app.get("/api/dq-values", response_model=AvailableDq)
    @app.get("/api/dq-values")
    async def get_element():
        return AVAILABLE_DQ  # {ion: {charge: {symmetry: {'initial': {'Dq': 0.1, ...}, 'final': {'Dq': 0.2, ...}}}}}

    @app.get("/api/config")
    async def get_element():
        try:
            quanty_path = get_quanty_path()
        except OSError:
            quanty_path = 'QUANTY NOT AVAILABLE'
        return {
            'beamline': get_beamline(),
            'visits': AVAILABLE_EXPIDS,
            'quanty_path': quanty_path,
            'available_dq_values': AVAILABLE_DQ,
        }

    @app.post("/api/scanfiles")
    async def scan_files(data: DataPath):
        if not os.path.isdir(data.path):
            logger.info('Path does not exist:', data.path)
            return {}
        filespec = get_path_filespec(data.path)
        logger.info(f"files in {data.path}: {filespec}")
        return filespec


    @app.post("/api/similar_scans")
    async def scan_files(data: DataPath):
        if not os.path.isfile(data.path):
            logger.info('File does not exist:', data.path)
            return {}
        measurements = find_similar_measurements(data.path)
        files = [m.filename for m in measurements]
        scan_numbers = [m.scan_number for m in measurements]
        logger.info(f"similar files to {data.path}: {files}")
        return {'files': files, 'scan_numbers': scan_numbers}

    @app.post("/api/simulation", response_model=SimulationOutputs)
    async def simulation(data: SimulationInputs):
        # Run Quanty
        logger.info('Now I run Quanty with the following parameters:\n', data)
        try:
            simulation = gen_simulation(
                ion=data.ion,
                ch_str=data.charge,
                symmetry=data.symmetry,
                beta=data.beta,
                dq=data.tenDq['10Dq_i'] if '10Dq_i' in data.tenDq else 0.0,
                mag_field=[data.bFieldX, data.bFieldY, data.bFieldZ],
                exchange_field=[data.hFieldX, data.hFieldY, data.hFieldZ],
                temperature=data.temperature,
                quanty_path=data.path,
            )
            logger.info(f"Running Quanty simulation: {simulation.label}")
            result = simulation.run_all()
            logger.debug(f"Simulation output: {result.stdout if result else 'None'}")
            logger.info(f"Analysing results of simulation: {simulation.label}")
            table, axis1, axis2 = simulation.analyse()
            data = {
                "message": f"simulation {simulation.label} succsefull", 
                "table": table, 
                "plot1": axis1, 
                "plot2": axis2
            }
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            data = {
                "message": f"Error running simulation: {e}",
                "table": f"Error running simulation: {e}",
                "plot1": {}, 
                "plot2": {},
            }
        packed_data = msgpack.packb(data, use_bin_type=True, default=encoder)
        return Response(content=packed_data, media_type="application/x-msgpack")


    @app.post("/api/simulations", response_model=SimulationOutputs)
    async def simulation(simulations: Simulations):
        # Run Quanty
        logger.info('Now I run Quanty with the following parameters:\n', simulations)
        try:
            simulation = gen_simulation(
                ion=simulations.ion,
                ch_str=simulations.charge,
                symmetry=simulations.symmetry,
                beta=simulations.beta,
                dq=simulations.tenDq['10Dq_i'] if '10Dq_i' in simulations.tenDq else 0.0,
                mag_field=[simulations.bFieldX, simulations.bFieldY, simulations.bFieldZ],
                exchange_field=[simulations.hFieldX, simulations.hFieldY, simulations.hFieldZ],
                temperature=simulations.temperature,
                quanty_path=simulations.path,
            )
            logger.info(f"Running Quanty simulation: {simulation.label}")
            result = simulation.run_all()
            logger.debug(f"Simulation output: {result.stdout if result else 'None'}")
            logger.info(f"Analysing results of simulation: {simulation.label}")
            table, axis1, axis2 = simulation.analyse()
            simulations = {
                "message": f"simulation {simulation.label} succsefull", 
                "table": table, 
                "plot1": axis1, 
                "plot2": axis2
            }
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            simulations = {
                "message": f"Error running simulation: {e}",
                "table": f"Error running simulation: {e}",
                "plot1": {}, 
                "plot2": {},
            }
        packed_data = msgpack.packb(simulations, use_bin_type=True, default=encoder)
        return Response(content=packed_data, media_type="application/x-msgpack")

    @app.post("/api/pol_pairs", response_model=list[lineProps])
    async def get_pairs(data: DataFiles):
        logger.info(f"Finding pairs in files: \n{'\n'.join(data.files)}")
        data = find_pairs(*data.files)
        logger.info(f"Found {len(data)} pairs")
        packed_data = msgpack.packb(data, use_bin_type=True, default=encoder)
        return Response(content=packed_data, media_type="application/x-msgpack")

    # @app.post("/api/measurement", response_model=MeasuredData)
    @app.post("/api/measurement")
    async def measurement(indata: LoadMeasuredData):
        logger.info(f"Finding pairs in files: \n{'\n'.join(indata.files)}")
        try:
            pol_set = find_pairs(*indata.files, background_type=indata.background_type)  # load files, check similarity, remove background and find pairs
            logger.info(f"Found {len(pol_set.measurements)} pairs")
            table = pol_set.table()
            data: MeasuredData = {
                'pol_pairs': [measurement.output() for measurement in pol_set.measurements],
                'average': pol_set.output(),
                'table': table,
                'element': pol_set.element,
                'field': [pol_set.field_x, pol_set.field_y, pol_set.field_z],
                'temperature': pol_set.temperature,
            }
        except ValueError as e:
            logger.error(f"Error finding pairs: {e}")
            data: MeasuredData = {
                'pol_pairs': [],
                'average': [],
                'table': f"Error finding pairs: {e}",
                'element': '',
                'field': [0, 0, 0],
                'temperature': 1.0,
            }
        packed_data = msgpack.packb(data, use_bin_type=True, default=encoder)
        return Response(content=packed_data, media_type="application/x-msgpack")

    @app.post("/api/metadata")
    async def metadata(indata: LoadMetadata):
        logger.info(f"Loading metadata: \n{indata.files}")
        meta_strings = {
            scn: gen_metadata_str(filename)
            for scn, filename in indata.files.items()
        }
        return meta_strings

    INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dist'))
    print(INDEX)
    logger.info(f'!!! Frontend: {INDEX}, ispath: {os.path.isdir(INDEX)}')
    app.mount('/', StaticFiles(directory=INDEX, html=True), 'frontend')
    return app


def polspex_api_server():
    import uvicorn
    import webbrowser
    # app = create_fastapi_app()
    webbrowser.open_new_tab('http://localhost:8123/')
    uvicorn.run('polspex.api:create_fastapi_app', host="0.0.0.0", port=8123, log_level="info", reload=True)


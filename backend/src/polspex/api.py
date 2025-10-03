import os 
import logging 
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import msgpack
import socket
import requests
import numpy as np

# import secrets
import webbrowser
from threading import Thread
from time import sleep
# from contextlib import asynccontextmanager
# import nest_asyncio
# import requests

from .environment import AVAILABLE_EXPIDS, get_path_filespec, get_beamline, get_quanty_path
from .parameters import AVAILABLE_SYMMETRIES, AVAILABLE_DQ
from .xas_analysis import find_pairs, gen_metadata_str, find_similar_measurements
from .plot_models import lineProps
from .quanty_runner import gen_simulation


# Generate a secure token
# jupyter_token = secrets.token_hex(32)

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
    from . import __version__
    
    app = FastAPI(title="PolSpeX FastAPI", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #@app.get("/api/elements", response_model=AvailableElements)
    @app.get("/api/elements")
    async def get_elements():
        return AVAILABLE_SYMMETRIES

    # @app.get("/api/dq-values", response_model=AvailableDq)
    @app.get("/api/dq-values")
    async def get_dqvalues():
        return AVAILABLE_DQ  # {ion: {charge: {symmetry: {'initial': {'Dq': 0.1, ...}, 'final': {'Dq': 0.2, ...}}}}}

    @app.get("/api/config")
    async def get_config():
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
    async def similar_scans(data: DataPath):
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
    async def simulations(simulations: Simulations):
        # Not in use!
        logger.info('Now I run Quanty with the following parameters:\n', simulations)
        for inputs in simulations.sims:
            try:
                simulation = gen_simulation(
                    ion=inputs.ion,
                    ch_str=inputs.charge,
                    symmetry=inputs.symmetry,
                    beta=inputs.beta,
                    dq=inputs.tenDq['10Dq_i'] if '10Dq_i' in inputs.tenDq else 0.0,
                    mag_field=[inputs.bFieldX, inputs.bFieldY, inputs.bFieldZ],
                    exchange_field=[inputs.hFieldX, inputs.hFieldY, inputs.hFieldZ],
                    temperature=inputs.temperature,
                    quanty_path=inputs.path,
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

    # deployment - dist in module directory
    INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dist'))
    if not os.path.isdir(INDEX):
        # dev mode - dist in monorepo frontend
        INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'frontend', 'dist'))
        print(f'!!! Dev Mode !!! running fronted from: {INDEX}')
    print(INDEX)
    logger.info(f'!!! Frontend: {INDEX}, ispath: {os.path.isdir(INDEX)}')
    app.mount('/', StaticFiles(directory=INDEX, html=True), 'frontend')
    return app


def is_port_free(port, host="0.0.0.0"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_server(host, port, timeout=30):
    url = f"http://{host}:{port}/"
    for _ in range(timeout * 2):  # check every 0.5s up to timeout seconds
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        sleep(0.5)
    return False


def start_browser_thread(port, host="localhost", timeout=30):
    """Opens the browser at localhost"""
    def _open_browser():
        sleep(1)  # wait for server to start
        if wait_for_server(host, port, timeout=timeout):
            webbrowser.open_new_tab(f'http://{host}:{port}/')
        else:
            print("Server did not start on time")
    th = Thread(target=_open_browser)
    th.start()


def polspex_api_server():
    import uvicorn

    HOST = "localhost"
    PORT = 8123
    max_tries = 10
    n_tries = 0
    while not is_port_free(PORT, HOST) and n_tries < max_tries:
        print(f"Port {PORT} is already in use")
        PORT += 1
        n_tries += 1

    if not is_port_free(PORT, HOST):
        raise Exception(f"Port {PORT} is already in use")
    
    # Prepare to run server
    start_browser_thread(PORT, HOST)

    # Start server
    uvicorn.run('polspex.api:create_fastapi_app', host=HOST, port=PORT, log_level="info", reload=True)



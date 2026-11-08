"""
Measured rooms from the BUT ReverbDB database for the reverberation of :func:`sdialog.audio.synthesis.to_audio`.

`BUT ReverbDB <https://speech.fit.vut.cz/software/but-speech-fit-reverb-database>`_ (Brno University of
Technology) contains impulse responses measured in 9 real rooms (offices, meeting and lecture rooms, hotel
rooms), from several loudspeaker positions to 31 microphones, together with recordings of each room's own
background noise at every microphone. SDialog does not include the database: ``ButReverbDB(path, download=True)``
downloads ``BUT_ReverbDB_rel_19_06_RIR-Only.tgz`` (8.7 GB, CC BY 4.0) from its website and extracts it into
``path`` the first time. It can also be downloaded by hand and extracted into a folder of its own (the archive has no
top folder), then given as ``path``:

    .. code-block:: bash

        wget http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz
        mkdir BUT_ReverbDB && tar xzf BUT_ReverbDB_rel_19_06_RIR-Only.tgz -C BUT_ReverbDB

Please cite it when publishing results obtained with it:

    I. Szöke, M. Skácel, L. Mošner, J. Paliesek and J. Černocký, "Building and Evaluation of a Real Room
    Impulse Response Dataset", IEEE Journal of Selected Topics in Signal Processing, 13(4), 863-876, 2019.

Example:

    .. code-block:: python

        from sdialog.audio import ButReverbDB

        but = ButReverbDB("BUT_ReverbDB", download=True)         # or rooms=["VUT_FIT_L207", ...]
        audio = dialog.to_audio(reverb=but, noise="room")        # both speakers in the same measured room,
                                                                 # with that room's own background noise
        # Only the caller in a measured room (e.g. a phone call), with its room noise at 10-20 dB SNR
        audio = dialog.to_audio(reverb={"Caller": but},
                                noise={"Caller": {"type": "room", "snr": (10, 20)}},
                                channel="g711")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@idiap.ch>, Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import logging
import tarfile
import urllib.request

import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import effects

logger = logging.getLogger(__name__)

URL = "http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"

# Layout of BUT ReverbDB: <room>/<MicIDxx>/<SpkIDxx_date_type>/<microphone 01-31>/{RIR,silence}/*.wav
_DEPTH = 4


class ButReverbDB:
    """
    Impulse responses and background noise measured in real rooms (BUT ReverbDB), to use as ``reverb`` in
    ``dialog.to_audio()``.

    For each dialogue, a room and a microphone are drawn at random (from the ``to_audio`` seed), and each
    speaker gets a different loudspeaker position in that room, so both speakers are recorded by the same
    microphone. Given per speaker (e.g. ``reverb={"Caller": but}``), each speaker gets a position of their own.
    With ``noise="room"``, the background noise is the one recorded by the same microphone, in the same room.

    :param path: Path to the extracted BUT ReverbDB (the folder containing the room folders, e.g. ``VUT_FIT_L207``).
    :type path: str
    :param rooms: Names of the rooms to use (default: all of them, see :attr:`rooms`).
    :type rooms: Optional[List[str]]
    :param download: If True and ``path`` does not exist, downloads the database (8.7 GB) and extracts it there.
    :type download: bool
    :raises FileNotFoundError: If no impulse responses with the BUT ReverbDB layout are found under ``path``.
    :raises ValueError: If a room in ``rooms`` is not in the database.

    Example:

        .. code-block:: python

            from sdialog.audio import ButReverbDB

            but = ButReverbDB("BUT_ReverbDB", rooms=["VUT_FIT_L207", "VUT_FIT_Q301"], download=True)
            print(but.rooms)
            audio = dialog.to_audio(reverb=but, noise="room", seed=0)
            audio.config["room"]   # room, microphone, positions and files used
    """

    def __init__(self, path: str, rooms: Optional[List[str]] = None, download: bool = False):
        self.path = os.path.abspath(path)
        if download and not os.path.isdir(self.path):
            self.download(self.path)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"BUT ReverbDB folder not found: '{path}' (use download=True to download it)")
        # room -> microphone -> loudspeaker position -> folder of the measurement
        positions: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for root, dirs, _ in os.walk(self.path):
            parts = os.path.relpath(root, self.path).split(os.sep)
            if len(parts) == _DEPTH and "RIR" in dirs:
                room, mic_array, speaker, mic = parts
                positions[room][f"{mic_array}/{mic}"][speaker] = root
            if len(parts) >= _DEPTH:
                dirs[:] = []  # do not walk into the RIR/silence folders
            dirs.sort()
        if not positions:
            raise FileNotFoundError(f"No BUT ReverbDB impulse responses found under '{path}' (expected "
                                    "<room>/<MicID>/<SpkID>/<microphone>/RIR/ folders)")
        if rooms is not None:
            unknown = [r for r in rooms if r not in positions]
            if unknown:
                raise ValueError(f"Unknown BUT ReverbDB room(s) {unknown}. Available: {sorted(positions)}")
            positions = {r: positions[r] for r in rooms}
        self._positions = {room: {mic: dict(spk) for mic, spk in mics.items()} for room, mics in positions.items()}

    @staticmethod
    def download(path: str, url: str = URL) -> str:
        """
        Downloads BUT ReverbDB (RIR-only release, 8.7 GB) and extracts it into ``path``. The archive is removed
        once extracted; an interrupted download or extraction leaves only ``.part`` files, so ``path`` exists only
        when complete.

        :param path: Folder to extract the database into (must not exist yet).
        :type path: str
        :param url: Address of the archive.
        :type url: str
        :return: ``path``.
        :rtype: str
        """
        path = os.path.abspath(path)
        if os.path.exists(path):
            raise FileExistsError(f"'{path}' already exists")
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        archive = path + ".tgz.part"
        progress = {"next": 0}

        def report(blocks, block_size, total):
            done = blocks * block_size
            if total > 0 and done >= progress["next"]:
                gb = 2 ** 30  # binary GB, as the 8.7 GB announced for the archive
                logger.info(f"BUT ReverbDB: downloaded {min(done, total) / gb:.1f} of {total / gb:.1f} GB")
                progress["next"] += total // 10

        logger.info(f"Downloading BUT ReverbDB (8.7 GB) from {url} ...")
        urllib.request.urlretrieve(url, archive, reporthook=report)
        logger.info(f"Extracting BUT ReverbDB into {path} ...")
        extract_to = path + ".part"
        with tarfile.open(archive) as tar:  # the archive has no top folder
            if hasattr(tarfile, "data_filter"):  # safe extraction where available (Python >= 3.12, recent 3.8-3.11)
                tar.extractall(extract_to, filter="data")
            else:
                tar.extractall(extract_to)
        os.replace(extract_to, path)
        os.remove(archive)
        return path

    @property
    def rooms(self) -> List[str]:
        """Names of the rooms in use."""
        return sorted(self._positions)

    def __len__(self) -> int:
        """Number of measured (loudspeaker, microphone) positions in use."""
        return sum(len(spk) for mics in self._positions.values() for spk in mics.values())

    def __repr__(self) -> str:
        return f"ButReverbDB(path={self.path!r}, rooms={len(self.rooms)}, positions={len(self)})"

    def json(self) -> dict:
        """Returns a JSON-serializable description (stored in the audio annotations)."""
        return {"type": "but", "path": self.path, "rooms": self.rooms}

    def compute_rirs(self,
                     n_sources: int,
                     sample_rate: int,
                     rng: Optional[np.random.Generator] = None) -> Tuple[List[np.ndarray], dict]:
        """
        Draws a room and a microphone, and loads the impulse response of ``n_sources`` different loudspeaker
        positions in that room (positions are reused only if the room has fewer than ``n_sources``).

        :param n_sources: Number of speakers to place.
        :type n_sources: int
        :param sample_rate: Sample rate of the returned impulse responses.
        :type sample_rate: int
        :param rng: Random generator (default: a new unseeded one).
        :type rng: Optional[np.random.Generator]
        :return: The impulse responses (one per speaker) and a description of the room and files used.
        :rtype: Tuple[List[np.ndarray], dict]
        """
        rng = rng or np.random.default_rng()
        room = self.rooms[int(rng.integers(len(self.rooms)))]
        mics = sorted(self._positions[room])
        mic = mics[int(rng.integers(len(mics)))]
        speakers = sorted(self._positions[room][mic])
        order = rng.permutation(len(speakers))
        chosen = [speakers[order[ix % len(speakers)]] for ix in range(n_sources)]
        rirs, rir_files = [], []
        for speaker in chosen:
            rir, file = effects.load_rir(os.path.join(self._positions[room][mic][speaker], "RIR"), rng, sample_rate)
            rirs.append(rir)
            rir_files.append(file)
        # Background noise recorded by this microphone (the same for every loudspeaker position)
        noise = None
        for speaker in chosen + speakers:
            folder = os.path.join(self._positions[room][mic][speaker], "silence")
            if os.path.isdir(folder):
                noise = folder
                break
        info = {"type": "but", "room": room, "microphone": mic, "speakers": chosen, "rir_files": rir_files,
                "noise": noise}
        return rirs, info

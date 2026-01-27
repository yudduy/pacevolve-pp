#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import numpy as np
import time
import mmh3
import random

class RandomJumpConsistentHashing():
    """
    One implementation of Random Jump Consistent Hashing (RJCH).

    Refactored to separate infrastructure/evaluation from core algorithms.
    """

    # ----------------------------------------------------------------------
    # INFRASTRUCTURE, STATE MANAGEMENT, AND EVALUATION METHODS
    # ----------------------------------------------------------------------

    def __init__(self, servers, duplicates, objects, epsilon):
        """
        servers - How many unique servers.
        duplicates - how many virtual copies of each server.
        objects - how many objects to put in.
        epsilon - (1 + epsilon) * objects / servers is the load cap for each server.
        """
        self.serversCount = servers
        self.totalServersAndPastCount = servers
        self.duplicatesCount = duplicates
        self.objectsCount = objects
        self.totalObjectsAndPastCount = objects
        self.epsilon = epsilon
        self.loadCap = int(np.ceil((1 + epsilon) * objects / servers))
        self.totalFull = 0

        self.servers = {i:0 for i in range(self.serversCount)} # Mapping from serverid: number of objects
        self.fullFlag = {i:False for i in range(self.serversCount)}
        self.serversToIdx = {} # Mapping of servers to the index in the long array.

        # Put the servers in a long array.
        self.serversArray = np.array([-1 for i in range(2**20)])
        self.serversArrayLen = len(self.serversArray)

        # Used to track what is contained in the server.
        self.serversContains = {i: set() for i in range(self.serversCount)}

        # Used to track all previous tries.
        self.objectsHistory = {i: [] for i in range(self.objectsCount)}

        # Where the object is now.
        self.objectsCurrent = {i: 0.5 for i in range(self.objectsCount)}

    def start(self):
        """
        Initializes and assigns objects to servers according to init values.
        """
        for i in self.servers:
            dups = []
            for _ in range(self.duplicatesCount):
                cur = np.random.randint(0, self.serversArrayLen)
                while self.serversArray[cur] != -1:
                    cur = np.random.randint(0, self.serversArrayLen)
                self.serversArray[cur] = i
                dups.append(cur)
            self.serversToIdx[i] = dups
        t = time.time()
        for i in range(self.objectsCount):
            if i == self.objectsCount - 1:
                t = time.time()
            self._add_one_object_algo(object_id=i)
        return time.time() - t


    def addOneObject(self):
        """
        Adds a single new object to the system.
        """
        object_id = self.totalObjectsAndPastCount
        history_list = self.objectsHistory[object_id] = []

        # Core algorithm to find a suitable server slot
        num = self._find_server_for_object_algo(history_list)
        curServer = self.serversArray[num]

        # State updates (Infra)
        self.servers[curServer] += 1
        if self.servers[curServer] == self.loadCap:
            self.fullFlag[curServer] = True
            self.totalFull += 1

        self.serversContains[curServer].add(object_id)
        self.objectsCurrent[object_id] = num
        self.objectsCount += 1
        self.totalObjectsAndPastCount += 1

    def removeOneObject(self):
        """
        Removes a randomly chosen object from the system.
        """
        if not self.objectsCurrent:
            return # No objects to remove

        # Get the object that this count refers to.
        objectNum = np.random.choice(list(self.objectsCurrent.keys()))

        # Get the server that object is in.
        serverIdx = self.objectsCurrent[objectNum]
        serverKey = self.serversArray[serverIdx]

        # State updates (Infra)
        self.serversContains[serverKey].remove(objectNum)
        self.servers[serverKey] -= 1
        self.objectsCount -= 1
        self.objectsCurrent.pop(objectNum)
        self.objectsHistory.pop(objectNum)

        wasFull = self.fullFlag[serverKey]
        if wasFull:
            self.fullFlag[serverKey] = False
            self.totalFull -= 1

            self._fillBinOne_algo(serverKey)

    def variance(self):
        """
        Returns current server load variance. (Eval metric)
        """
        return np.var(list(self.servers.values()))

    def pctOfFullBins(self):
        """
        Returns the pct of full bins. (Eval metric)
        """
        return sum(self.fullFlag.values()) / len(self.fullFlag) if self.fullFlag else 0

    def timeAddOneObject(self):
        """
        Tries to add an object and returns the wall time
        """
        t = time.time()
        self._assign_object_algo()
        return time.time() - t
        

    def assignObjectTotalSteps(self):
        """
        Simulates throwing another object in and seeing how many
        total steps it needs to try before finding a non-full one. (Eval simulation)
        """
        num = np.random.randint(0, self.serversArrayLen)
        counter = 1
        while self.serversArray[num] == -1 or self.fullFlag[self.serversArray[num]]:
            num = np.random.randint(0, self.serversArrayLen)
            counter += 1
        return counter
    
    def _fillBinOne_algo(self, serverKey):
        """
        Tries to refill a specific server (serverKey) that is no longer full.
        This is the core rebalancing algorithm.
        """
        if self.servers[serverKey] >= self.loadCap:
            return

        dups = self.serversToIdx.get(serverKey, [])
        if not dups:
            return
        np.random.shuffle(dups)

        lstObjects = list(self.objectsHistory.keys())
        np.random.shuffle(lstObjects)

        for object_id in lstObjects:
            if object_id not in self.objectsCurrent: continue # Object was removed

            current_server_idx = self.objectsCurrent[object_id]
            current_server_key = self.serversArray[current_server_idx]

            if current_server_key == serverKey:
                continue # Object is already on this server

            history = self.objectsHistory[object_id]
            for target_server_idx in dups:
                if target_server_idx in history:
                    idx_in_history = history.index(target_server_idx)

                    # Move the object
                    # Remove from old server
                    self.serversContains[current_server_key].remove(object_id)
                    self.servers[current_server_key] -= 1
                    wasFullOld = self.fullFlag[current_server_key]
                    if wasFullOld:
                        self.fullFlag[current_server_key] = False
                        self.totalFull -= 1

                    # Add to new server (serverKey)
                    self.serversContains[serverKey].add(object_id)
                    self.servers[serverKey] += 1
                    self.objectsCurrent[object_id] = target_server_idx

                    # Truncate history
                    self.objectsHistory[object_id] = history[:idx_in_history + 1]

                    # Recursive call on the server that lost an object
                    if wasFullOld:
                         self._fillBinOne_algo(current_server_key)

                    # Check if the target server is now full
                    if self.servers[serverKey] == self.loadCap:
                        if not self.fullFlag[serverKey]:
                            self.totalFull += 1
                            self.fullFlag[serverKey] = True
                        return # This server is full, stop trying to fill it

                    # Found a replacement for this server, break from inner dups loop
                    goto_next_object = True
                    break
            else: # No break from dups loop
                goto_next_object = False

            if goto_next_object:
                 pass # Continue to the next object

    # ----------------------------------------------------------------------
    # CORE ALGORITHM METHODS
    # ----------------------------------------------------------------------
    # RegexTagCustomPruningAlgorithmStart
    def _add_one_object_algo(self, object_id):
        """
        Places an object using a linear probe with a single, conditional
        "look-behind" check to mitigate cascading overflow.
        """
        history_list = self.objectsHistory.setdefault(object_id, [])

        # Hash object to a starting position in serversArray
        obj_hash_input = f"object_{object_id}"
        start_index = mmh3.hash(obj_hash_input, seed=object_id) % self.serversArrayLen

        found_slot = False
        has_looked_behind = False # Flag to ensure we only look behind once

        # Search circularly from the start_index for the next slot with a server
        for k in range(self.serversArrayLen):
            current_index = (start_index + k) % self.serversArrayLen
            history_list.append(current_index)

            server_id = self.serversArray[current_index]

            if server_id != -1:  # Found a server
                if not self.fullFlag[server_id]:  # Check if the server is not full
                    # Assign object to this server
                    self.serversContains[server_id].add(object_id)
                    self.servers[server_id] += 1
                    if self.servers[server_id] == self.loadCap:
                        self.fullFlag[server_id] = True
                        self.totalFull += 1
                    self.objectsCurrent[object_id] = current_index
                    found_slot = True
                    break  # Object placed
                else:  # Server is full, this is where we might look behind
                    if not has_looked_behind:
                        has_looked_behind = True # Prevent future look-behinds

                        # Perform a single check at the counter-clockwise position
                        look_behind_index = (current_index - 1 + self.serversArrayLen) % self.serversArrayLen
                        history_list.append(look_behind_index) # Record the probe
                        
                        behind_server_id = self.serversArray[look_behind_index]

                        if behind_server_id != -1 and not self.fullFlag[behind_server_id]:
                            # Found an available server behind, place it here
                            self.serversContains[behind_server_id].add(object_id)
                            self.servers[behind_server_id] += 1
                            if self.servers[behind_server_id] == self.loadCap:
                                self.fullFlag[behind_server_id] = True
                                self.totalFull += 1
                            self.objectsCurrent[object_id] = look_behind_index
                            found_slot = True
                            break # Object placed
                    # If look-behind failed or was already done, just continue the linear probe
            # else: slot is empty, continue scanning
        # End of linear scan loop

        if not found_slot:
             # This would mean all servers are full or no server slots were found,
             raise RuntimeError(f"Failed to place object {object_id}: No available server found in serversArray.")
    # RegexTagCustomPruningAlgorithmEnd
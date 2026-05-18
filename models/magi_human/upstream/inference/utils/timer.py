# Copyright (c) 2026 SandAI. All Rights Reserved.
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

from datetime import datetime
from typing import Callable

import torch

from .logger import print_rank_0


class EventPathTimer:
    """
    A lightweight class for recording time without any distributed barrier.

    This class allows for recording elapsed time between events without requiring
    synchronization across distributed processes. It maintains the previous message
    and time to calculate the duration between consecutive records.
    """

    def __init__(self):
        """
        Initialize the EventPathTimer.

        This constructor sets the previous message and time to None, preparing
        the instance for recording events.
        """
        self.prev_message: str = None
        self.prev_time: datetime = None

    def reset(self):
        """
        Reset the recorded message and time.

        This method clears the previous message and time, allowing for a fresh
        start in recording new events.
        """
        self.prev_message = None
        self.prev_time = None

    def synced_record(self, message, print_fn: Callable[[str], None] = print_rank_0):
        """
        Record the current time with a message.

        Args:
            message (str): A message to log along with the current time.

        This method synchronizes the CUDA operations, records the current time,
        and calculates the elapsed time since the last recorded message, if any.
        It then logs the elapsed time along with the previous and current messages.
        """
        torch.cuda.synchronize()
        current_time = datetime.now()
        if self.prev_message is not None:
            print_fn(
                f"\nTime Elapsed: [{current_time - self.prev_time}] From [{self.prev_message} ({self.prev_time})] To [{message} ({current_time})]"
            )
        self.prev_message = message
        self.prev_time = current_time


_GLOBAL_LIGHT_TIMER = EventPathTimer()


def event_path_timer() -> EventPathTimer:
    """Get the current EventPathTimer instance.

    Returns:
        EventPathTimer: The current EventPathTimer instance.

    Raises:
        AssertionError: If the EventPathTimer has not been initialized.
    """
    assert _GLOBAL_LIGHT_TIMER is not None, "light time recorder is not initialized"
    return _GLOBAL_LIGHT_TIMER

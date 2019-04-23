#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
Main file of the project.
"""

def __main():
    from experiments import train_keras, test_keras, train_pytorch, test_pytorch

    # to train Timeception using keras
    train_keras.__main()

    # or using pytorch
    # train_pytorch.__main()

    # to test Timeception using keras
    # test_keras.__main()

    # or using pytorch
    # test_pytorch.__main()

if __name__ == '__main__':
    __main()
    pass

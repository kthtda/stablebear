#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

print("!!!!!!!!!! Printing wheel contents !!!!!!!!!!")

import glob
import os
import sys


def list_package(pkgname):
    site_packages_dirs = [
        p for p in sys.path if os.path.isdir(p) and "site-packages" in p
    ]

    found = False
    for sp in site_packages_dirs:
        pkg_candidates = glob.glob(os.path.join(sp, f"{pkgname}*"))
        for pkg in pkg_candidates:
            print(f"Contents of installed package/module at {pkg}:")
            if os.path.isdir(pkg):
                for dp, dn, filenames in os.walk(pkg):
                    for f in filenames:
                        print(os.path.join(dp, f))
            else:
                # Compiled extension (pyd/so)
                print(pkg)
            found = True

    if not found:
        print("Cannot find stablebear in site-packages")


list_package("stablebear")
list_package("stablebear_cpu")
list_package("_sb_cpp")
list_package("stablebear-cpu")

"""
vasp_analysis 完整测试套件

测试目录：
  Relax : /data2/home/luodh/Test/VaspAnalysis-Test/Relax-Test
  DOS   : /data2/home/luodh/Test/VaspAnalysis-Test/Doscar-Text
  COHP  : /data2/home/luodh/Test/VaspAnalysis-Test/Lobster-Text

运行方式：
  # 运行全部测试
  python test_vasp_analysis.py

  # 运行单个模块
  python test_vasp_analysis.py RelaxTest
  python test_vasp_analysis.py DosTest
  python test_vasp_analysis.py CohpTest
"""

import sys
import json
import unittest
import traceback
from pathlib import Path

# ── 路径配置（按需修改）──────────────────────────────────
RELAX_DIR  = Path("/data2/home/luodh/Test/VaspAnalysis-Test/Relax-Test")
DOS_DIR    = Path("/data2/home/luodh/Test/VaspAnalysis-Test/Doscar-Text")
LOBSTER_DIR= Path("/data2/home/luodh/Test/VaspAnalysis-Test/Lobster-Text")
# ────────────────────────────────────────────────────────

# 将项目根目录加入 sys.path（按实际位置调整）
sys.path.insert(0, "/data2/home/luodh/Git-workflow/Workflow_for_vasp_catalysis")
from vasp_analysis_mod.Analysis import RelaxAnalysis, DosAnalysis, DoscarParser
from vasp_analysis_mod.Analysis import CohpAnalysis, dispatch, register_task, ApiResponse, VaspAnalysisBase


# ============================================================
# 辅助函数
# ============================================================

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_json_preview(json_str: str, max_list_items: int = 3):
    """
    打印 JSON 预览：列表类型的字段只显示前 N 条，避免刷屏。
    """
    def truncate(obj, depth=0):
        if isinstance(obj, list):
            preview = [truncate(i, depth+1) for i in obj[:max_list_items]]
            if len(obj) > max_list_items:
                preview.append(f"... ({len(obj)} items total)")
            return preview
        elif isinstance(obj, dict):
            return {k: truncate(v, depth+1) for k, v in obj.items()}
        return obj

    try:
        data = json.loads(json_str)
        print(json.dumps(truncate(data), indent=2, ensure_ascii=False))
    except Exception:
        print(json_str[:500])

def assert_response_ok(response_json: str, context: str = "") -> dict:
    """断言 ApiResponse 成功，返回 data 字典"""
    resp = json.loads(response_json)
    assert resp["success"], (
        f"[FAIL] {context}\n"
        f"  code   : {resp.get('code')}\n"
        f"  message: {resp.get('message')}"
    )
    return resp["data"]


# ============================================================
# 1. Relax 测试
# ============================================================

class RelaxTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print_section("Relax Analysis Tests")
        if not RELAX_DIR.exists():
            raise unittest.SkipTest(f"Relax dir not found: {RELAX_DIR}")
        cls.analyzer = RelaxAnalysis(work_dir=RELAX_DIR, save_data=False)

    # ── 文件存在性检查 ────────────────────────────────────

    def test_01_required_files_exist(self):
        """检查至少存在一个可读取的输出文件"""
        files = ["OSZICAR", "OUTCAR", "vasprun.xml"]
        found = [f for f in files if (RELAX_DIR / f).exists()]
        print(f"\n  [Files Found] {found}")
        self.assertTrue(len(found) > 0, "No VASP output files found in Relax dir.")

    # ── 核心属性测试 ──────────────────────────────────────

    def test_02_final_energy(self):
        """最终能量应为有效负数"""
        energy = self.analyzer._get_final_energy()
        print(f"\n  Final Energy : {energy:.6f} eV")
        self.assertIsNotNone(energy, "Final energy is None.")
        self.assertLess(energy, 0, f"Energy {energy} should be negative.")

    def test_03_fermi_level(self):
        """费米能级应在合理范围内"""
        fermi = self.analyzer._get_fermi()
        print(f"\n  Fermi Level  : {fermi:.4f} eV")
        self.assertIsNotNone(fermi)
        self.assertTrue(-30 < fermi < 30, f"Fermi level {fermi} out of range.")

    def test_04_ionic_steps(self):
        """离子步数应 >= 1"""
        steps = self.analyzer._get_ionic_steps()
        elec  = self.analyzer._get_elec_steps()
        print(f"\n  Ionic Steps  : {steps}")
        print(f"  Elec Steps   : {elec}")
        self.assertGreaterEqual(steps, 1)

    def test_05_convergence(self):
        """收敛状态检查（仅打印，不强制断言）"""
        converged, last_dE = self.analyzer._get_convergence()
        print(f"\n  Converged    : {converged}")
        print(f"  Last dE      : {last_dE}")
        # 不强制要求收敛，但打印警告
        if not converged:
            print("  [Warning] Calculation did NOT converge!")

    def test_06_magnetization(self):
        """磁矩读取（非磁性体系跳过）"""
        total_mag = self.analyzer._get_total_mag()
        site_mag  = self.analyzer._get_site_mag()
        print(f"\n  Total Mag    : {total_mag}")
        print(f"  Site Mag     : {len(site_mag)} sites")
        if site_mag:
            print(f"  First Site   : {site_mag[0]}")
        # 非磁性体系 total_mag 可能为 None，不强制断言

    def test_07_nelect(self):
        """总电子数应为正数"""
        nelect = self.analyzer._get_nelect()
        print(f"\n  Total Elec   : {nelect}")
        if nelect is not None:
            self.assertGreater(nelect, 0)

    # ── analyze() 完整流程 ────────────────────────────────

    def test_08_analyze_full(self):
        """完整 analyze() 返回有效 ApiResponse"""
        resp_json = self.analyzer.analyze().to_json()
        data = assert_response_ok(resp_json, "RelaxAnalysis.analyze()")
        print("\n  [analyze() Result Preview]")
        print_json_preview(resp_json)

        # 关键字段验证
        self.assertIn("converged",        data)
        self.assertIn("final_energy_eV",  data)
        self.assertIn("fermi_level_eV",   data)
        self.assertIn("ionic_steps",      data)
        self.assertIn("warnings",         data)

    # ── dispatch 接口测试 ─────────────────────────────────

    def test_09_dispatch_relax(self):
        """通过 dispatch 调用 relax 任务"""
        resp_json = dispatch("relax", RELAX_DIR)
        data = assert_response_ok(resp_json, "dispatch('relax')")
        print(f"\n  [dispatch] converged={data['converged']}, "
              f"energy={data['final_energy_eV']:.4f} eV")


# ============================================================
# 2. DOS 测试
# ============================================================

class DosTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print_section("DOS Analysis Tests")
        if not DOS_DIR.exists():
            raise unittest.SkipTest(f"DOS dir not found: {DOS_DIR}")
        cls.analyzer = DosAnalysis(work_dir=DOS_DIR, save_data=False)

    # ── DoscarParser 底层测试 ─────────────────────────────

    def test_01_doscar_exists(self):
        """DOSCAR 文件必须存在"""
        doscar = DOS_DIR / "DOSCAR"
        self.assertTrue(doscar.exists(), f"DOSCAR not found in {DOS_DIR}")
        print(f"\n  DOSCAR size  : {doscar.stat().st_size / 1024:.1f} KB")

    def test_02_parser_header(self):
        """DoscarParser 正确解析文件头"""
        parser = DoscarParser(DOS_DIR / "DOSCAR")
        print(f"\n  NIONS        : {parser.nions}")
        print(f"  NEDOS        : {parser.nedos}")
        print(f"  Efermi       : {parser.efermi:.4f} eV")
        print(f"  ISPIN        : {parser.ispin}")
        self.assertGreater(parser.nions,  0)
        self.assertGreater(parser.nedos,  0)
        self.assertTrue(-30 < parser.efermi < 30)
        self.assertIn(parser.ispin, [1, 2])

    def test_03_parser_tdos_shape(self):
        """TDOS 数组形状正确"""
        parser = DoscarParser(DOS_DIR / "DOSCAR")
        tdos   = parser.tdos
        print(f"\n  TDOS shape   : {tdos.shape}")
        self.assertEqual(tdos.shape[0], parser.nedos)
        # 非自旋 3 列，自旋 5 列
        expected_cols = 5 if parser.ispin == 2 else 3
        self.assertEqual(tdos.shape[1], expected_cols,
                         f"Expected {expected_cols} cols, got {tdos.shape[1]}")

    def test_04_parser_pdos_count(self):
        """PDOS 块数量与 NIONS 一致"""
        parser = DoscarParser(DOS_DIR / "DOSCAR")
        pdos   = parser.pdos
        print(f"\n  PDOS blocks  : {len(pdos)} (expected {parser.nions})")
        print(f"  PDOS col names: {parser.get_pdos_col_names()}")
        self.assertEqual(len(pdos), parser.nions)

    def test_05_structure_load(self):
        """结构从 CONTCAR/POSCAR 正确加载"""
        structure = self.analyzer.structure
        elements  = self.analyzer.site_elements
        print(f"\n  Structure    : {structure.formula}")
        print(f"  N sites      : {len(structure)}")
        print(f"  Elements     : {list(dict.fromkeys(elements))}")
        self.assertEqual(len(elements), self.analyzer.parser.nions,
                         "Site count mismatch between structure and DOSCAR.")

    def test_06_efermi_source(self):
        """费米能级读取（OUTCAR 优先）"""
        efermi = self.analyzer.efermi
        print(f"\n  Efermi       : {efermi:.4f} eV")
        self.assertTrue(-30 < efermi < 30)

    # ── TDOS 测试 ─────────────────────────────────────────

    def test_07_total_dos(self):
        """TDOS 提取正确"""
        df = self.analyzer.get_total_dos()
        print(f"\n  TDOS shape   : {df.shape}")
        print(f"  TDOS columns : {list(df.columns)}")
        print(f"  Energy range : [{df['energy_eV'].min():.2f}, "
              f"{df['energy_eV'].max():.2f}] eV")
        self.assertIn("energy_eV", df.columns)
        self.assertIn("tdos_up",   df.columns)
        self.assertGreater(len(df), 0)

    def test_08_alignment_fermi(self):
        """费米能级对齐后能量零点正确"""
        self.analyzer.set_alignment("fermi")
        df = self.analyzer.get_total_dos()
        efermi = self.analyzer.efermi
        # 对齐后，能量轴应包含 0（费米能级附近）
        self.assertTrue(
            df["energy_eV"].min() < 0 < df["energy_eV"].max(),
            "After Fermi alignment, energy axis should span across 0."
        )
        print(f"\n  After alignment: E range = "
              f"[{df['energy_eV'].min():.2f}, {df['energy_eV'].max():.2f}] eV")

    # ── PDOS 测试 ─────────────────────────────────────────

    def test_09_site_dos(self):
        """位点 0 的总 DOS 提取"""
        df = self.analyzer.get_site_dos(site_index=0)
        el = self.analyzer.site_elements[0]
        print(f"\n  Site 0 ({el}) DOS shape : {df.shape}")
        print(f"  Columns : {list(df.columns)}")
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

    def test_10_site_spd_dos(self):
        """位点 0 的 SPD-DOS 提取"""
        df = self.analyzer.get_site_spd_dos(site_index=0)
        el = self.analyzer.site_elements[0]
        print(f"\n  Site 0 ({el}) SPD-DOS columns : {list(df.columns) if df is not None else 'None'}")
        self.assertIsNotNone(df, "SPD-DOS should not be None (check LORBIT setting).")

    def test_11_element_spd_dos(self):
        """元素汇总 SPD-DOS（所有元素）"""
        df = self.analyzer.get_element_spd_dos()
        print(f"\n  Element SPD-DOS shape   : {df.shape if df is not None else 'None'}")
        print(f"  Element SPD-DOS columns : {list(df.columns) if df is not None else 'None'}")
        self.assertIsNotNone(df)

    def test_12_site_t2g_eg_dos(self):
        """
        t2g/eg 分解 DOS（仅对含 d 轨道的位点有效）。
        自动寻找第一个过渡金属位点进行测试。
        """
        TRANSITION_METALS = {
            "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
            "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
            "Hf","Ta","W","Re","Os","Ir","Pt","Au",
        }
        tm_sites = [
            i for i, el in enumerate(self.analyzer.site_elements)
            if el in TRANSITION_METALS
        ]
        if not tm_sites:
            self.skipTest("No transition metal sites found for t2g/eg test.")

        site_idx = tm_sites[0]
        el       = self.analyzer.site_elements[site_idx]
        df       = self.analyzer.get_site_t2g_eg_dos(site_index=site_idx)
        print(f"\n  t2g/eg site  : {site_idx} ({el})")
        print(f"  t2g/eg cols  : {list(df.columns) if df is not None else 'None'}")
        self.assertIsNotNone(df, f"t2g/eg DOS is None for site {site_idx} ({el}).")

    def test_13_orbital_statistics(self):
        """d-band center 等轨道统计量"""
        all_elements = list(dict.fromkeys(self.analyzer.site_elements))
        print(f"\n  Elements in system: {all_elements}")

        for el in all_elements:
            stats = self.analyzer.get_orbital_statistics(
                element=el, orbital="d", erange=[-10, 5]
            )
            if stats:
                print(f"\n  [{el}] d-band statistics:")
                for k, v in stats.items():
                    print(f"    {k:<35} = {v:.4f}")
            else:
                print(f"\n  [{el}] No d-band data (may not have d electrons).")

    # ── analyze() 完整流程 ────────────────────────────────

    def test_14_analyze_full(self):
        """完整 analyze() 返回有效 ApiResponse"""
        resp_json = self.analyzer.analyze(orbital="d", erange=[-10, 5]).to_json()
        data = assert_response_ok(resp_json, "DosAnalysis.analyze()")
        print("\n  [analyze() Result Preview]")
        print_json_preview(resp_json)

        self.assertIn("is_spin_polarized",  data)
        self.assertIn("fermi_level_eV",     data)
        self.assertIn("total_dos",          data)
        self.assertIn("element_spd_dos",    data)
        self.assertIn("orbital_statistics", data)

    # ── dispatch 接口测试 ─────────────────────────────────

    def test_15_dispatch_dos(self):
        """通过 dispatch 调用 dos 任务"""
        resp_json = dispatch("dos", DOS_DIR, orbital="d")
        data = assert_response_ok(resp_json, "dispatch('dos')")
        print(f"\n  [dispatch] spin={data['is_spin_polarized']}, "
              f"efermi={data['fermi_level_eV']:.4f} eV, "
              f"n_ions={data['n_ions']}")

    # ── save_data 测试 ────────────────────────────────────

    def test_16_save_csv(self):
        """save_data=True 时正确输出 CSV 文件"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = DosAnalysis(
                work_dir=DOS_DIR,
                save_data=True,
                output_dir=tmpdir,
            )
            analyzer.get_total_dos(save=True)
            analyzer.get_element_spd_dos(save=True)

            saved = list(Path(tmpdir).glob("*.csv"))
            print(f"\n  Saved CSV files: {[f.name for f in saved]}")
            self.assertGreater(len(saved), 0, "No CSV files were saved.")


# ============================================================
# 3. COHP 测试
# ============================================================

class CohpTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print_section("COHP Analysis Tests")
        if not LOBSTER_DIR.exists():
            raise unittest.SkipTest(f"Lobster dir not found: {LOBSTER_DIR}")
        cls.analyzer = CohpAnalysis(work_dir=LOBSTER_DIR, save_data=False)

    # ── 文件存在性 ────────────────────────────────────────

    def test_01_lobster_files_exist(self):
        """LOBSTER 输出文件必须存在"""
        for fname in ["COHPCAR.lobster", "ICOHPLIST.lobster"]:
            path = LOBSTER_DIR / fname
            exists = path.exists()
            size   = f"{path.stat().st_size / 1024:.1f} KB" if exists else "N/A"
            print(f"\n  {fname:<25} : {'✓' if exists else '✗'}  {size}")
            self.assertTrue(exists, f"{fname} not found in {LOBSTER_DIR}")

    # ── ICOHPLIST 测试 ────────────────────────────────────

    def test_02_icohplist_load(self):
        """ICOHPLIST 正确加载"""
        icohplist = self.analyzer.icohplist
        self.assertIsNotNone(icohplist, "ICOHPLIST.lobster failed to load.")
        n_bonds = len(icohplist.icohplist)
        print(f"\n  Total bonds in ICOHPLIST : {n_bonds}")
        self.assertGreater(n_bonds, 0)

    def test_03_icohp_summary_default(self):
        """ICOHP 汇总表（前 10 个键）"""
        df = self.analyzer.get_icohp_summary(n_top=10)
        print(f"\n  ICOHP Summary (top 10):\n{df.to_string(index=False)}")
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        self.assertIn("bond_label", df.columns)
        self.assertIn("icohp_eV",   df.columns)
        self.assertIn("length_Ang", df.columns)
        # ICOHP 值通常为负（成键）
        bonding = df[df["icohp_eV"] < 0]
        print(f"\n  Bonding interactions (ICOHP < 0): {len(bonding)}")

    def test_04_icohp_top1_strongest(self):
        """最强键的 |ICOHP| 应最大"""
        df = self.analyzer.get_icohp_summary(n_top=5)
        if len(df) >= 2:
            self.assertGreaterEqual(
                abs(df.iloc[0]["icohp_eV"]),
                abs(df.iloc[1]["icohp_eV"]),
                "Results should be sorted by |ICOHP| descending."
            )
        print(f"\n  Strongest bond : {df.iloc[0]['bond_label']} "
              f"({df.iloc[0]['atom1']}-{df.iloc[0]['atom2']}) "
              f"ICOHP = {df.iloc[0]['icohp_eV']:.4f} eV")

    def test_05_icohp_element_pair_filter(self):
        """
        元素对过滤功能（自动从 ICOHPLIST 中推断存在的元素对）。
        """
        df_all = self.analyzer.get_icohp_summary(n_top=999)
        if df_all is None or df_all.empty:
            self.skipTest("No ICOHP data available.")

        # 从数据中提取所有出现的元素对
        def get_element(atom_str):
            return "".join(c for c in atom_str if c.isalpha())

        pairs = set()
        for _, row in df_all.iterrows():
            el1 = get_element(row["atom1"])
            el2 = get_element(row["atom2"])
            pairs.add(tuple(sorted([el1, el2])))

        print(f"\n  Detected element pairs: {pairs}")
        test_pair = list(pairs)[0]

        df_filtered = self.analyzer.get_icohp_summary(
            n_top=999, element_pair=test_pair
        )
        print(f"  Filter by {test_pair}: {len(df_filtered)} bonds")
        self.assertGreater(len(df_filtered), 0)

        # 验证过滤结果中只包含目标元素对
        for _, row in df_filtered.iterrows():
            el1 = get_element(row["atom1"])
            el2 = get_element(row["atom2"])
            self.assertEqual({el1, el2}, set(test_pair))

    # ── COHPCAR 测试 ──────────────────────────────────────

    def test_06_cohpcar_load(self):
        """COHPCAR 正确加载"""
        cohpcar = self.analyzer.cohpcar
        self.assertIsNotNone(cohpcar, "COHPCAR.lobster failed to load.")
        n_bonds = len(cohpcar.cohp_data)
        print(f"\n  Total bonds in COHPCAR : {n_bonds}")
        print(f"  Bond labels (first 5)  : {list(cohpcar.cohp_data.keys())[:5]}")
        self.assertGreater(n_bonds, 0)

    def test_07_cohp_curves_all(self):
        """提取全部 COHP 曲线"""
        df = self.analyzer.get_cohp_curves()
        print(f"\n  COHP curves shape   : {df.shape}")
        print(f"  COHP curves columns : {list(df.columns)[:6]} ...")
        self.assertIsNotNone(df)
        self.assertIn("energy_eV", df.columns)
        self.assertGreater(df.shape[1], 1)   # 至少有一个键的数据

    def test_08_cohp_curves_erange(self):
        """能量范围截取功能"""
        erange = [-10.0, 5.0]
        df = self.analyzer.get_cohp_curves(erange=erange)
        print(f"\n  COHP erange={erange}: {len(df)} points")
        self.assertTrue(df["energy_eV"].min() >= erange[0] - 0.01)
        self.assertTrue(df["energy_eV"].max() <= erange[1] + 0.01)

    def test_09_cohp_curves_specific_bond(self):
        """指定键对标签提取 COHP 曲线"""
        cohpcar     = self.analyzer.cohpcar
        first_label = list(cohpcar.cohp_data.keys())[0]
        df = self.analyzer.get_cohp_curves(bond_labels=[first_label])
        print(f"\n  Bond '{first_label}' COHP shape : {df.shape}")
        self.assertIn(f"{first_label}_up", df.columns)

    def test_10_integrated_cohp(self):
        """数值积分 COHP（占据态）"""
        cohpcar     = self.analyzer.cohpcar
        first_label = list(cohpcar.cohp_data.keys())[0]
        result = self.analyzer.get_integrated_cohp(
            bond_label=first_label,
            erange=[-20.0, 0.0],   # 积分到费米能级
        )
        print(f"\n  Integrated COHP for '{first_label}':")
        for k, v in result.items():
            print(f"    {k:<20} = {v:.4f} eV")
        self.assertIn("icohp_total", result)

    # ── analyze() 完整流程 ────────────────────────────────

    def test_11_analyze_full(self):
        """完整 analyze() 返回有效 ApiResponse"""
        resp_json = self.analyzer.analyze(n_top_bonds=5, erange=[-10, 5]).to_json()
        data = assert_response_ok(resp_json, "CohpAnalysis.analyze()")
        print("\n  [analyze() Result Preview]")
        print_json_preview(resp_json)

        self.assertIn("icohp_summary", data)
        self.assertIn("cohp_curves",   data)
        self.assertIn("n_bonds",       data)
        self.assertGreater(data["n_bonds"], 0)

    # ── dispatch 接口测试 ─────────────────────────────────

    def test_12_dispatch_cohp(self):
        """通过 dispatch 调用 cohp 任务"""
        resp_json = dispatch("cohp", LOBSTER_DIR, n_top_bonds=5)
        data = assert_response_ok(resp_json, "dispatch('cohp')")
        print(f"\n  [dispatch] n_bonds={data['n_bonds']}")
        if data["icohp_summary"]:
            top = data["icohp_summary"][0]
            print(f"  Strongest bond : {top['bond_label']} "
                  f"ICOHP={top['icohp_eV']:.4f} eV")


# ============================================================
# 4. Dispatcher 通用测试
# ============================================================

class DispatcherTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print_section("Dispatcher Tests")

    def test_01_unknown_task_type(self):
        """未知任务类型返回 400"""
        resp = json.loads(dispatch("unknown_task", RELAX_DIR))
        print(f"\n  Unknown task response: code={resp['code']}, msg={resp['message']}")
        self.assertFalse(resp["success"])
        self.assertEqual(resp["code"], 400)

    def test_02_nonexistent_dir(self):
        """不存在的目录返回 404"""
        resp = json.loads(dispatch("relax", "/nonexistent/path/xyz"))
        print(f"\n  Bad path response: code={resp['code']}, msg={resp['message']}")
        self.assertFalse(resp["success"])
        self.assertIn(resp["code"], [404, 500])

    def test_03_register_custom_task(self):
        """自定义任务注册与调用"""
        class DummyAnalysis(VaspAnalysisBase):
            def analyze(self, **kwargs):
                return ApiResponse.ok(
                    data={"custom": True, "work_dir": str(self.work_dir)},
                    message="Dummy task complete"
                )

        register_task("dummy", DummyAnalysis)
        resp = json.loads(dispatch("dummy", RELAX_DIR))
        print(f"\n  Custom task response: {resp['message']}")
        self.assertTrue(resp["success"])
        self.assertTrue(resp["data"]["custom"])

    def test_04_invalid_register(self):
        """注册不含 analyze() 方法的类应报错"""
        class BadAnalysis:
            pass
        with self.assertRaises(TypeError):
            register_task("bad", BadAnalysis)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 支持命令行指定测试类
    # python test_vasp_analysis.py RelaxTest
    # python test_vasp_analysis.py DosTest
    # python test_vasp_analysis.py CohpTest
    # python test_vasp_analysis.py          ← 运行全部

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    if len(sys.argv) > 1:
        target = sys.argv[1]
        cls_map = {
            "RelaxTest":      RelaxTest,
            "DosTest":        DosTest,
            "CohpTest":       CohpTest,
            "DispatcherTest": DispatcherTest,
        }
        if target not in cls_map:
            print(f"Unknown test class '{target}'. "
                  f"Available: {list(cls_map.keys())}")
            sys.exit(1)
        suite.addTests(loader.loadTestsFromTestCase(cls_map[target]))
    else:
        for cls in [RelaxTest, DosTest, CohpTest, DispatcherTest]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
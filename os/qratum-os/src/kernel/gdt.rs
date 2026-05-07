//! Global Descriptor Table + TSS (Agent 2, Agent 6).
//!
//! Layout:
//!   index 1 → kernel code      (CS, ring 0, 64-bit)
//!   index 2 → kernel data      (DS/SS, ring 0)
//!   index 3 → user data        (DS/SS, ring 3)         [user data BEFORE code on x86_64]
//!   index 4 → user code        (CS, ring 3, 64-bit)
//!   index 5 → TSS (16 bytes, two slots)
//!
//! IST 0  →  double-fault stack (separate, never re-entered)
//! RSP0   →  kernel stack used on ring-3 → ring-0 transitions

#![allow(static_mut_refs, dead_code)]

use x86_64::{
    instructions::tables::load_tss,
    registers::segmentation::{Segment, CS, DS, ES, FS, GS, SS},
    structures::{
        gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector},
        tss::TaskStateSegment,
    },
    VirtAddr,
};

pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;
pub const PAGE_FAULT_IST_INDEX:   u16 = 1;

const STACK_SIZE: usize = 16 * 1024;

#[repr(align(16))]
struct Stack([u8; STACK_SIZE]);

static mut DF_STACK:  Stack = Stack([0; STACK_SIZE]);
static mut PF_STACK:  Stack = Stack([0; STACK_SIZE]);
static mut RSP0_STACK: Stack = Stack([0; STACK_SIZE]);

static mut TSS: TaskStateSegment = TaskStateSegment::new();

pub struct Selectors {
    pub kernel_code: SegmentSelector,
    pub kernel_data: SegmentSelector,
    pub user_code:   SegmentSelector,
    pub user_data:   SegmentSelector,
    pub tss:         SegmentSelector,
}

static mut GDT: Option<(GlobalDescriptorTable, Selectors)> = None;

/// Initialise GDT + TSS and load the new selectors.
///
/// MUST run after ExitBootServices (we own CR3/CS/SS now). Idempotent —
/// safe to call once at boot only.
pub fn init() {
    unsafe {
        // ---- TSS stacks ----------------------------------------------------
        let df_top = VirtAddr::from_ptr(&raw const DF_STACK)  + STACK_SIZE as u64;
        let pf_top = VirtAddr::from_ptr(&raw const PF_STACK)  + STACK_SIZE as u64;
        let rsp0   = VirtAddr::from_ptr(&raw const RSP0_STACK) + STACK_SIZE as u64;

        TSS.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = df_top;
        TSS.interrupt_stack_table[PAGE_FAULT_IST_INDEX   as usize] = pf_top;
        TSS.privilege_stack_table[0] = rsp0;

        // ---- GDT -----------------------------------------------------------
        let mut gdt = GlobalDescriptorTable::new();
        let kernel_code = gdt.append(Descriptor::kernel_code_segment());
        let kernel_data = gdt.append(Descriptor::kernel_data_segment());
        let user_data   = gdt.append(Descriptor::user_data_segment());
        let user_code   = gdt.append(Descriptor::user_code_segment());
        let tss         = gdt.append(Descriptor::tss_segment(&*(&raw const TSS)));

        GDT = Some((
            gdt,
            Selectors { kernel_code, kernel_data, user_code, user_data, tss },
        ));

        let (gdt_ref, sel) = GDT.as_ref().unwrap();
        gdt_ref.load();

        // Reload segments. Data segments → kernel_data; CS → kernel_code via far return.
        CS::set_reg(sel.kernel_code);
        DS::set_reg(sel.kernel_data);
        ES::set_reg(sel.kernel_data);
        FS::set_reg(sel.kernel_data);
        GS::set_reg(sel.kernel_data);
        SS::set_reg(sel.kernel_data);

        load_tss(sel.tss);
    }
}

pub fn selectors() -> &'static Selectors {
    unsafe {
        GDT.as_ref().map(|(_, s)| s).expect("gdt::init() must run before selectors()")
    }
}

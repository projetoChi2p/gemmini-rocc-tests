include $(abs_top_srcdir)/Makefrag

tests = \
	test_mlp_quant \
	test_matmul_quant \
	test_matmul_quant2 \
	test_layernorm_quant \
	test_softmax_quant \
	test_gelu_quant \
	test_resadd_quant \
	test_embedding_quant \
	test_ffn_quant \
	test_attention_quant \
	test_classifier_head_quant \
	test_full_model_quant \
	
tests_baremetal = $(tests:=-baremetal)
runs_baremetal = $(addsuffix .run,$(tests_baremetal))
ifdef BAREMETAL_ONLY
	tests_linux =
	tests_pk =
else
	tests_linux = $(tests:=-linux)
	tests_pk = $(tests:=-pk)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_nn.h $(abs_top_srcdir)/include/gemmini_testutils.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux) $(tests_pk)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$^

junk += $(tests_baremetal) $(tests_linux) $(tests_pk)


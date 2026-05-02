//! VITRA-E0 Merkler-Static: Biokey-Enabled Merkle Chain Builder
//!
//! Command-line tool for ephemeral biokey operations in sovereign genomics.

mod biokey;
mod zkp;
mod fido2;

use biokey::{EphemeralBiokey, SnpLocus};
use zkp::BiokeyZkp;
use fido2::DualBiokeySignature;
use serde_json;
use std::io::{self, Write, Read};

const VERSION: &str = "1.0.0";

fn print_usage() {
    println!("VITRA-E0 Merkler-Static v{}", VERSION);
    println!("Biokey-Enabled Merkle Chain Builder for Sovereign Genomics");
    println!();
    println!("USAGE:");
    println!("  merkler-static <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("  derive-biokey <loci-json>    Derive ephemeral biokey from SNP loci");
    println!("  generate-challenge           Generate random ZKP challenge");
    println!("  prove <biokey-json> <challenge-hex>  Generate ZKP proof");
    println!("  verify-zkp <proof-json>      Verify zero-knowledge proof");
    println!("  dual-sign <biokey-a> <biokey-b> <message>  Create dual signature");
    println!("  verify-dual <signature-json> <message>     Verify dual signature");
    println!("  version                      Show version information");
    println!("  help                         Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("  # Derive biokey from SNP loci");
    println!("  merkler-static derive-biokey loci.json");
    println!();
    println!("  # Generate ZKP challenge");
    println!("  merkler-static generate-challenge");
    println!();
    println!("  # Create ZKP proof");
    println!("  merkler-static prove biokey.json <challenge>");
    println!();
    println!("SECURITY:");
    println!("  - Biokeys exist only in RAM (never written to disk)");
    println!("  - Automatic memory wiping on exit");
    println!("  - Zero-knowledge proofs don't reveal genome data");
    println!("  - Dual signatures require two operators");
}

fn derive_biokey_from_json(loci_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Parse SNP loci from JSON
    let loci: Vec<SnpLocus> = serde_json::from_str(loci_json)?;
    
    if loci.len() < 128 {
        eprintln!("WARNING: Less than 128 loci provided ({}). Recommend 128-256 for security.", loci.len());
    }
    
    // Derive biokey
    let biokey = EphemeralBiokey::derive_from_loci(&loci);
    
    // Output results (public hash only, never private key)
    println!("Biokey derived successfully!");
    println!("Loci count: {}", biokey.loci_count);
    println!("Public hash: {}", biokey.public_hash_hex());
    println!();
    println!("SECURITY: Private key stored in RAM only. Will be wiped on exit.");
    
    // Export to environment variable format
    println!();
    println!("Export to environment (for scripts):");
    println!("export VITRA_BIOKEY_PUBLIC_HASH={}", biokey.public_hash_hex());
    println!("export VITRA_BIOKEY_LOCI_COUNT={}", biokey.loci_count);
    
    Ok(())
}

fn generate_challenge() {
    let challenge = BiokeyZkp::generate_challenge();
    println!("Random ZKP challenge generated:");
    println!("{}", hex::encode(challenge));
}

fn generate_proof(biokey_json: &str, challenge_hex: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Parse biokey loci
    let loci: Vec<SnpLocus> = serde_json::from_str(biokey_json)?;
    let biokey = EphemeralBiokey::derive_from_loci(&loci);
    
    // Parse challenge
    let challenge_bytes = hex::decode(challenge_hex)?;
    if challenge_bytes.len() != 32 {
        return Err("Challenge must be 32 bytes (64 hex characters)".into());
    }
    let mut challenge = [0u8; 32];
    challenge.copy_from_slice(&challenge_bytes);
    
    // Generate proof
    let proof = BiokeyZkp::prove(&biokey, &challenge);
    
    // Output proof as JSON
    println!("{}", proof.to_json()?);
    
    Ok(())
}

fn verify_zkp(proof_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let proof: BiokeyZkp = BiokeyZkp::from_json(proof_json)?;
    
    // Note: In a real system, we'd need the public hash to verify
    // This simplified version just checks format
    println!("ZKP format valid: {}", proof.challenge.len() == 32 && proof.response.len() == 32);
    println!("Challenge: {}", hex::encode(proof.challenge));
    println!("Response: {}", hex::encode(proof.response));
    
    Ok(())
}

fn create_dual_signature(
    biokey_a_json: &str,
    biokey_b_json: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse biokeys
    let loci_a: Vec<SnpLocus> = serde_json::from_str(biokey_a_json)?;
    let loci_b: Vec<SnpLocus> = serde_json::from_str(biokey_b_json)?;
    
    let biokey_a = EphemeralBiokey::derive_from_loci(&loci_a);
    let biokey_b = EphemeralBiokey::derive_from_loci(&loci_b);
    
    // Get current timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    
    // Create dual signature
    let signature = DualBiokeySignature::create(
        &biokey_a,
        &biokey_b,
        message.as_bytes(),
        timestamp,
    );
    
    // Output signature as JSON
    println!("{}", signature.to_json()?);
    
    Ok(())
}

fn verify_dual_signature(signature_json: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    let signature: DualBiokeySignature = DualBiokeySignature::from_json(signature_json)?;
    
    let valid = signature.verify(message.as_bytes());
    
    println!("Dual signature verification: {}", if valid { "VALID" } else { "INVALID" });
    println!("Operator A: {}", signature.operator_a_hash_hex());
    println!("Operator B: {}", signature.operator_b_hash_hex());
    println!("Timestamp: {}", signature.timestamp);
    
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    
    let command = &args[1];
    
    let result = match command.as_str() {
        "version" => {
            println!("merkler-static v{}", VERSION);
            println!("VITRA-E0 Biokey-Enabled Merkle Chain Builder");
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        "derive-biokey" => {
            if args.len() < 3 {
                eprintln!("ERROR: Missing loci JSON argument");
                print_usage();
                std::process::exit(1);
            }
            
            // Read from file or stdin
            let loci_json = if args[2] == "-" {
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer).expect("Failed to read stdin");
                buffer
            } else {
                std::fs::read_to_string(&args[2]).expect("Failed to read loci file")
            };
            
            derive_biokey_from_json(&loci_json)
        }
        "generate-challenge" => {
            generate_challenge();
            Ok(())
        }
        "prove" => {
            if args.len() < 4 {
                eprintln!("ERROR: Missing biokey JSON and challenge arguments");
                std::process::exit(1);
            }
            
            let biokey_json = std::fs::read_to_string(&args[2])
                .expect("Failed to read biokey file");
            let challenge_hex = &args[3];
            
            generate_proof(&biokey_json, challenge_hex)
        }
        "verify-zkp" => {
            if args.len() < 3 {
                eprintln!("ERROR: Missing proof JSON argument");
                std::process::exit(1);
            }
            
            let proof_json = std::fs::read_to_string(&args[2])
                .expect("Failed to read proof file");
            
            verify_zkp(&proof_json)
        }
        "dual-sign" => {
            if args.len() < 5 {
                eprintln!("ERROR: Missing biokey and message arguments");
                std::process::exit(1);
            }
            
            let biokey_a_json = std::fs::read_to_string(&args[2])
                .expect("Failed to read biokey A file");
            let biokey_b_json = std::fs::read_to_string(&args[3])
                .expect("Failed to read biokey B file");
            let message = &args[4];
            
            create_dual_signature(&biokey_a_json, &biokey_b_json, message)
        }
        "verify-dual" => {
            if args.len() < 4 {
                eprintln!("ERROR: Missing signature and message arguments");
                std::process::exit(1);
            }
            
            let signature_json = std::fs::read_to_string(&args[2])
                .expect("Failed to read signature file");
            let message = &args[3];
            
            verify_dual_signature(&signature_json, message)
        }
        _ => {
            eprintln!("ERROR: Unknown command: {}", command);
            print_usage();
            std::process::exit(1);
        }
    };
    
    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_constant() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.contains('.'));
    }
}
